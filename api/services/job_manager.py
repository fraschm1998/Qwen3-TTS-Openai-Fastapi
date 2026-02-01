# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Async job manager for batch TTS generation.

Disk (state.json) is the source of truth. The in-memory dict is a runtime cache.
Supports chapter-level checkpointing, two-queue priority (interactive vs batch),
reliable webhook delivery, and graceful cancellation.
"""

import asyncio
import dataclasses
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiofiles
import numpy as np

from ..structures.job_schemas import (
    ChapterProgress,
    ChapterStatus,
    JobCreateRequest,
    JobListResponse,
    JobResponse,
    JobStatus,
    JobSummary,
    WebhookPayload,
)
from .audio_encoding import encode_audio
from .text_processing import normalize_text
from .tts_engine import CancelledError, generate_speech_chunked

logger = logging.getLogger(__name__)

MAX_WEBHOOK_ATTEMPTS = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Internal state dataclasses (not exposed via API)
# ---------------------------------------------------------------------------


@dataclass
class ChapterState:
    index: int
    title: Optional[str]
    text: str
    status: str = ChapterStatus.PENDING.value
    error: Optional[str] = None
    audio_filename: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class JobState:
    id: str
    status: str  # JobStatus value
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    model: str = "qwen3-tts"
    voice: str = "Vivian"
    language: Optional[str] = "Auto"
    instruct: Optional[str] = None
    speed: float = 1.0
    response_format: str = "mp3"
    chapters: list[ChapterState] = field(default_factory=list)
    webhook_url: Optional[str] = None
    webhook_attempts: int = 0
    webhook_delivered: bool = False
    cancel_requested: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _job_to_response(job: JobState) -> JobResponse:
    chapters = [
        ChapterProgress(
            index=ch.index,
            title=ch.title,
            status=ChapterStatus(ch.status),
            error=ch.error,
            audio_filename=ch.audio_filename,
            duration_seconds=ch.duration_seconds,
        )
        for ch in job.chapters
    ]
    completed = sum(1 for ch in job.chapters if ch.status == ChapterStatus.COMPLETED.value)
    current = next(
        (ch.index + 1 for ch in job.chapters if ch.status == ChapterStatus.PROCESSING.value),
        None,
    )
    return JobResponse(
        id=job.id,
        status=JobStatus(job.status),
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        model=job.model,
        voice=job.voice,
        language=job.language,
        response_format=job.response_format,
        total_chapters=len(job.chapters),
        completed_chapters=completed,
        current_chapter=current,
        chapters=chapters,
        webhook_url=job.webhook_url,
        webhook_delivered=job.webhook_delivered if job.webhook_url else None,
        error=job.error,
    )


def _job_to_summary(job: JobState) -> JobSummary:
    completed = sum(1 for ch in job.chapters if ch.status == ChapterStatus.COMPLETED.value)
    current = next(
        (ch.index + 1 for ch in job.chapters if ch.status == ChapterStatus.PROCESSING.value),
        None,
    )
    return JobSummary(
        id=job.id,
        status=JobStatus(job.status),
        created_at=job.created_at,
        updated_at=job.updated_at,
        total_chapters=len(job.chapters),
        completed_chapters=completed,
        current_chapter=current,
    )


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_job_manager: Optional["JobManager"] = None


def get_job_manager() -> "JobManager":
    if _job_manager is None:
        raise RuntimeError("JobManager not initialized")
    return _job_manager


def set_job_manager(manager: "JobManager") -> None:
    global _job_manager
    _job_manager = manager


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------


class JobManager:
    def __init__(self, output_dir: str = "output/jobs") -> None:
        self._jobs: dict[str, JobState] = {}
        self._batch_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._priority_queue: asyncio.Queue[tuple[asyncio.Future, dict[str, Any]]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._output_dir = Path(output_dir)
        self._state_file = self._output_dir / "state.json"
        self._http_client: Any = None  # httpx.AsyncClient, lazily typed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        await self._recover_state()

        try:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30)
        except ImportError:
            logger.warning("httpx not installed — webhook delivery disabled")

        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"Job queue started. Output: {self._output_dir}")

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        # Sentinels to unblock queues
        await self._batch_queue.put(None)
        try:
            dummy_future: asyncio.Future = asyncio.get_event_loop().create_future()
            dummy_future.cancel()
            await self._priority_queue.put((dummy_future, {}))
        except Exception:
            pass

        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=30)
            except asyncio.TimeoutError:
                logger.warning("Worker did not stop within 30 s — cancelling")
                self._worker_task.cancel()

        await self._persist_state()

        if self._http_client:
            await self._http_client.aclose()

        logger.info("Job queue stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_job(self, request: JobCreateRequest) -> JobResponse:
        job_id = str(uuid.uuid4())
        now = _now_iso()

        chapters = [
            ChapterState(index=i, title=ch.title, text=ch.text)
            for i, ch in enumerate(request.chapters)
        ]

        job = JobState(
            id=job_id,
            status=JobStatus.PENDING.value,
            created_at=now,
            updated_at=now,
            model=request.model,
            voice=request.voice,
            language=request.language,
            instruct=request.instruct,
            speed=request.speed,
            response_format=request.response_format,
            chapters=chapters,
            webhook_url=request.webhook_url,
        )

        job_dir = self._output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        self._jobs[job_id] = job
        await self._persist_state()
        await self._batch_queue.put(job_id)

        logger.info(f"Job {job_id} submitted with {len(chapters)} chapter(s)")
        return _job_to_response(job)

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return _job_to_response(job)

    def list_jobs(self) -> JobListResponse:
        summaries = sorted(
            (_job_to_summary(j) for j in self._jobs.values()),
            key=lambda s: s.created_at,
            reverse=True,
        )
        return JobListResponse(jobs=summaries, total=len(summaries))

    async def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False

        if job.status in (
            JobStatus.COMPLETED.value,
            JobStatus.COMPLETED_WITH_ERRORS.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            return False

        job.cancel_requested = True

        # If still pending (not yet picked up by worker), cancel immediately
        if job.status == JobStatus.PENDING.value:
            job.status = JobStatus.CANCELLED.value
            job.updated_at = _now_iso()
            for ch in job.chapters:
                if ch.status == ChapterStatus.PENDING.value:
                    ch.status = ChapterStatus.SKIPPED.value
            await self._persist_state()

        return True

    async def delete_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False

        # Cancel first if still running
        if job.status in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
            await self.cancel_job(job_id)

        del self._jobs[job_id]

        job_dir = self._output_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

        await self._persist_state()
        return True

    def get_chapter_audio_path(self, job_id: str, chapter_index: int) -> Optional[Path]:
        """Return the audio file path for a chapter (1-indexed)."""
        job = self._jobs.get(job_id)
        if job is None:
            return None

        idx = chapter_index - 1  # convert to 0-indexed
        if idx < 0 or idx >= len(job.chapters):
            return None

        ch = job.chapters[idx]
        if not ch.audio_filename:
            return None

        path = self._output_dir / job_id / ch.audio_filename
        if not path.exists():
            return None

        return path

    async def generate_priority(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Submit an interactive TTS request on the priority queue.

        The caller awaits the returned future, which the worker fulfills
        ahead of any batch work.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[np.ndarray, int]] = loop.create_future()
        params = dict(text=text, voice=voice, language=language, instruct=instruct, speed=speed)
        await self._priority_queue.put((future, params))
        return await future

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        logger.info("Job worker started")
        while not self._shutdown_event.is_set():
            # High-priority first (non-blocking)
            try:
                future, params = self._priority_queue.get_nowait()
                if not future.cancelled():
                    await self._process_priority(future, params)
                continue
            except asyncio.QueueEmpty:
                pass

            # Low-priority batch (blocking with timeout)
            try:
                job_id = await asyncio.wait_for(self._batch_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if job_id is None:
                break

            await self._process_job(job_id)

        logger.info("Job worker stopped")

    async def _process_priority(
        self, future: asyncio.Future, params: dict[str, Any]
    ) -> None:
        try:
            result = await generate_speech_chunked(**params)
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)

    async def _process_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.cancel_requested:
            return

        job.status = JobStatus.PROCESSING.value
        job.updated_at = _now_iso()
        await self._persist_state()

        try:
            for chapter in job.chapters:
                # Checkpoint: skip already-completed chapters (restart safety)
                audio_path = self._output_dir / job_id / (chapter.audio_filename or "")
                if (
                    chapter.status == ChapterStatus.COMPLETED.value
                    and chapter.audio_filename
                    and audio_path.exists()
                ):
                    logger.debug(f"Job {job_id}: skipping completed chapter {chapter.index + 1}")
                    continue

                # Cancel check between chapters
                if job.cancel_requested:
                    self._mark_remaining_skipped(job, chapter.index)
                    break

                chapter.status = ChapterStatus.PROCESSING.value
                job.updated_at = _now_iso()
                await self._persist_state()

                try:
                    normalized = normalize_text(chapter.text)
                    audio, sr = await generate_speech_chunked(
                        text=normalized,
                        voice=job.voice,
                        language=job.language or "Auto",
                        instruct=job.instruct,
                        speed=job.speed,
                        cancel_check=lambda: job.cancel_requested,
                    )

                    # Cancel check after generation
                    if job.cancel_requested:
                        self._mark_remaining_skipped(job, chapter.index)
                        break

                    filename = f"chapter_{chapter.index + 1:04d}.{job.response_format}"
                    filepath = self._output_dir / job_id / filename
                    audio_bytes = encode_audio(audio, job.response_format, sr)

                    async with aiofiles.open(filepath, "wb") as f:
                        await f.write(audio_bytes)

                    chapter.status = ChapterStatus.COMPLETED.value
                    chapter.audio_filename = filename
                    chapter.duration_seconds = round(len(audio) / sr, 2)

                except CancelledError:
                    self._mark_remaining_skipped(job, chapter.index)
                    break
                except Exception as e:
                    logger.error(f"Job {job_id} chapter {chapter.index + 1} failed: {e}")
                    chapter.status = ChapterStatus.FAILED.value
                    chapter.error = str(e)

                job.updated_at = _now_iso()
                await self._persist_state()

                # Yield to priority queue between chapters
                await self._drain_priority_queue()

        except Exception as e:
            logger.error(f"Job {job_id} processing error: {e}")
            job.error = str(e)

        # Determine final status
        self._resolve_job_status(job)
        await self._persist_state()

        # Webhook
        if job.webhook_url and not job.webhook_delivered:
            await self._fire_webhook(job)

    def _mark_remaining_skipped(self, job: JobState, from_index: int) -> None:
        job.status = JobStatus.CANCELLED.value
        for ch in job.chapters:
            if ch.index >= from_index and ch.status in (
                ChapterStatus.PENDING.value,
                ChapterStatus.PROCESSING.value,
            ):
                ch.status = ChapterStatus.SKIPPED.value
        job.completed_at = _now_iso()
        job.updated_at = job.completed_at

    def _resolve_job_status(self, job: JobState) -> None:
        if job.status == JobStatus.CANCELLED.value:
            return

        failed = sum(1 for ch in job.chapters if ch.status == ChapterStatus.FAILED.value)
        total = len(job.chapters)

        if failed == total:
            job.status = JobStatus.FAILED.value
            job.error = "All chapters failed"
        elif failed > 0:
            job.status = JobStatus.COMPLETED_WITH_ERRORS.value
            job.error = f"{failed} of {total} chapters failed"
        else:
            job.status = JobStatus.COMPLETED.value

        job.completed_at = _now_iso()
        job.updated_at = job.completed_at

    async def _drain_priority_queue(self) -> None:
        while True:
            try:
                future, params = self._priority_queue.get_nowait()
                if not future.cancelled():
                    await self._process_priority(future, params)
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Webhook
    # ------------------------------------------------------------------

    async def _fire_webhook(self, job: JobState) -> None:
        if not self._http_client:
            return

        failed = sum(1 for ch in job.chapters if ch.status == ChapterStatus.FAILED.value)
        completed = sum(1 for ch in job.chapters if ch.status == ChapterStatus.COMPLETED.value)

        payload = WebhookPayload(
            job_id=job.id,
            status=JobStatus(job.status),
            completed_at=job.completed_at,
            total_chapters=len(job.chapters),
            completed_chapters=completed,
            failed_chapters=failed,
            error=job.error,
        )

        try:
            resp = await self._http_client.post(
                job.webhook_url,
                json=payload.model_dump(),
                timeout=10.0,
            )
            job.webhook_delivered = resp.is_success
        except Exception as e:
            logger.error(f"Webhook delivery failed for job {job.id}: {e}")
            job.webhook_delivered = False

        job.webhook_attempts += 1
        job.updated_at = _now_iso()
        await self._persist_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_state(self) -> None:
        data = {
            job_id: dataclasses.asdict(job)
            for job_id, job in self._jobs.items()
        }
        tmp = self._state_file.with_suffix(".tmp")
        try:
            async with aiofiles.open(tmp, "w") as f:
                await f.write(json.dumps(data, indent=2))
            os.replace(str(tmp), str(self._state_file))
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    async def _recover_state(self) -> None:
        if not self._state_file.exists():
            return

        try:
            async with aiofiles.open(self._state_file, "r") as f:
                raw = await f.read()
            data = json.loads(raw)
        except Exception as e:
            logger.error(f"Failed to recover state: {e}")
            return

        for job_id, job_dict in data.items():
            chapters = [
                ChapterState(**ch) for ch in job_dict.pop("chapters", [])
            ]
            job = JobState(**job_dict, chapters=chapters)
            self._jobs[job_id] = job

            # Re-enqueue incomplete jobs
            if job.status in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
                job.status = JobStatus.PENDING.value
                # Reset any PROCESSING chapters back to PENDING (they didn't finish)
                for ch in job.chapters:
                    if ch.status == ChapterStatus.PROCESSING.value:
                        ch.status = ChapterStatus.PENDING.value
                await self._batch_queue.put(job_id)
                logger.info(f"Re-enqueued incomplete job {job_id}")

            # Retry undelivered webhooks for terminal jobs
            if (
                job.webhook_url
                and not job.webhook_delivered
                and job.webhook_attempts < MAX_WEBHOOK_ATTEMPTS
                and job.status in (
                    JobStatus.COMPLETED.value,
                    JobStatus.COMPLETED_WITH_ERRORS.value,
                    JobStatus.FAILED.value,
                )
            ):
                asyncio.get_event_loop().call_soon(
                    lambda j=job: asyncio.ensure_future(self._fire_webhook(j))
                )

        logger.info(f"Recovered {len(self._jobs)} job(s) from state.json")
