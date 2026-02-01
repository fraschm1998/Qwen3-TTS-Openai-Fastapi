# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the async job queue system.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from api.backends import factory
from api.services.job_manager import JobManager
from api.structures.job_schemas import (
    ChapterInput,
    ChapterStatus,
    JobCreateRequest,
    JobStatus,
)


@pytest.fixture
def client():
    """Create a test client (triggers lifespan startup/shutdown)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def mock_backend():
    """Mock the TTS backend for all tests."""
    mock = MagicMock()
    mock.is_ready.return_value = True
    mock.get_backend_name.return_value = "mock"
    mock.get_model_id.return_value = "mock-tts"
    mock.get_device_info.return_value = {"gpu_available": False, "device": "cpu"}
    mock.get_supported_voices.return_value = ["Vivian"]
    mock.get_supported_languages.return_value = ["English"]
    mock.supports_voice_cloning.return_value = False
    mock.generate_speech = AsyncMock(
        return_value=(np.zeros(2400, dtype=np.float32), 24000)
    )
    factory._backend_instance = mock
    yield mock
    factory._backend_instance = None


def _poll_job(client, job_id: str, terminal_statuses=None, max_wait: float = 10.0):
    """Poll until job reaches a terminal status."""
    if terminal_statuses is None:
        terminal_statuses = {"completed", "completed_with_errors", "failed", "cancelled"}
    start = time.time()
    while time.time() - start < max_wait:
        resp = client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in terminal_statuses:
            return data
        time.sleep(0.2)
    raise TimeoutError(f"Job {job_id} did not reach terminal status within {max_wait}s")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestJobSchemas:

    def test_requires_input_or_chapters(self):
        with pytest.raises(ValidationError, match="input.*chapters"):
            JobCreateRequest()

    def test_rejects_both_input_and_chapters(self):
        with pytest.raises(ValidationError, match="not both"):
            JobCreateRequest(
                input="hello",
                chapters=[ChapterInput(text="world")],
            )

    def test_single_input_converts_to_chapters(self):
        req = JobCreateRequest(input="hello world")
        assert req.input is None
        assert len(req.chapters) == 1
        assert req.chapters[0].text == "hello world"

    def test_chapters_accepted(self):
        req = JobCreateRequest(
            chapters=[
                ChapterInput(title="Ch 1", text="First chapter"),
                ChapterInput(text="Second chapter"),
            ]
        )
        assert len(req.chapters) == 2
        assert req.chapters[0].title == "Ch 1"

    def test_default_tts_params(self):
        req = JobCreateRequest(input="test")
        assert req.voice == "Vivian"
        assert req.response_format == "mp3"
        assert req.speed == 1.0


# ---------------------------------------------------------------------------
# Job CRUD endpoints
# ---------------------------------------------------------------------------


class TestJobEndpoints:

    def test_create_job_single_input(self, client):
        resp = client.post("/v1/jobs", json={"input": "Hello world"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert data["total_chapters"] == 1
        assert len(data["id"]) > 0

    def test_create_job_batch(self, client):
        resp = client.post(
            "/v1/jobs",
            json={
                "chapters": [
                    {"title": "Chapter 1", "text": "First chapter text."},
                    {"title": "Chapter 2", "text": "Second chapter text."},
                    {"text": "Third chapter without title."},
                ],
                "voice": "Ryan",
                "response_format": "wav",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_chapters"] == 3
        assert data["voice"] == "Ryan"
        assert data["response_format"] == "wav"
        assert data["chapters"][0]["title"] == "Chapter 1"

    def test_create_job_rejects_empty(self, client):
        resp = client.post("/v1/jobs", json={})
        assert resp.status_code == 422

    def test_list_jobs(self, client):
        client.post("/v1/jobs", json={"input": "Job one"})
        client.post("/v1/jobs", json={"input": "Job two"})

        resp = client.get("/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        assert len(data["jobs"]) >= 2

    def test_get_job(self, client):
        create_resp = client.post("/v1/jobs", json={"input": "Test"})
        job_id = create_resp.json()["id"]

        resp = client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

    def test_get_nonexistent_job(self, client):
        resp = client.get("/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_job(self, client):
        create_resp = client.post("/v1/jobs", json={"input": "Delete me"})
        job_id = create_resp.json()["id"]

        resp = client.delete(f"/v1/jobs/{job_id}")
        assert resp.status_code == 204

        resp = client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_job(self, client):
        resp = client.delete("/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_audio_download_before_ready(self, client):
        create_resp = client.post("/v1/jobs", json={"input": "Not ready"})
        job_id = create_resp.json()["id"]

        resp = client.get(f"/v1/jobs/{job_id}/audio/1")
        assert resp.status_code == 404

    def test_audio_download_invalid_chapter(self, client):
        create_resp = client.post("/v1/jobs", json={"input": "One chapter"})
        job_id = create_resp.json()["id"]

        resp = client.get(f"/v1/jobs/{job_id}/audio/99")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------------


class TestJobProcessing:

    def test_single_chapter_completes(self, client):
        resp = client.post("/v1/jobs", json={"input": "Short text."})
        job_id = resp.json()["id"]

        data = _poll_job(client, job_id)
        assert data["status"] == "completed"
        assert data["completed_chapters"] == 1

    def test_multi_chapter_completes(self, client):
        resp = client.post(
            "/v1/jobs",
            json={
                "chapters": [
                    {"text": "Chapter one."},
                    {"text": "Chapter two."},
                    {"text": "Chapter three."},
                ]
            },
        )
        job_id = resp.json()["id"]

        data = _poll_job(client, job_id)
        assert data["status"] == "completed"
        assert data["completed_chapters"] == 3

    def test_chapter_audio_downloadable(self, client):
        resp = client.post("/v1/jobs", json={"input": "Download me."})
        job_id = resp.json()["id"]

        _poll_job(client, job_id)

        audio_resp = client.get(f"/v1/jobs/{job_id}/audio/1")
        assert audio_resp.status_code == 200
        assert len(audio_resp.content) > 0

    def test_failed_chapter_does_not_stop_job(self, client, mock_backend):
        call_count = 0

        async def _generate_with_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated failure")
            return (np.zeros(2400, dtype=np.float32), 24000)

        mock_backend.generate_speech = AsyncMock(side_effect=_generate_with_failure)

        resp = client.post(
            "/v1/jobs",
            json={
                "chapters": [
                    {"text": "Chapter one."},
                    {"text": "Chapter two (will fail)."},
                    {"text": "Chapter three."},
                ]
            },
        )
        job_id = resp.json()["id"]

        data = _poll_job(client, job_id)
        assert data["status"] == "completed_with_errors"
        assert data["chapters"][1]["status"] == "failed"
        assert data["chapters"][0]["status"] == "completed"
        assert data["chapters"][2]["status"] == "completed"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestJobPersistence:

    def test_state_file_created(self, client):
        from api.services.job_manager import get_job_manager

        manager = get_job_manager()
        client.post("/v1/jobs", json={"input": "Persist me."})

        assert manager._state_file.exists()

        with open(manager._state_file) as f:
            data = json.load(f)
        assert len(data) >= 1

    def test_completed_job_persisted(self, client):
        from api.services.job_manager import get_job_manager

        resp = client.post("/v1/jobs", json={"input": "Test persist."})
        job_id = resp.json()["id"]

        _poll_job(client, job_id)

        manager = get_job_manager()
        with open(manager._state_file) as f:
            data = json.load(f)

        assert job_id in data
        assert data[job_id]["status"] == "completed"


# ---------------------------------------------------------------------------
# Resume after restart
# ---------------------------------------------------------------------------


class TestJobResume:

    def test_resume_skips_completed_chapters(self, client, mock_backend):
        """Create a job, let 2/4 chapters complete, recover, verify skips."""
        from api.services.job_manager import get_job_manager

        # Track how many times generate_speech is called
        call_count = 0
        original_generate = mock_backend.generate_speech

        async def counting_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return (np.zeros(2400, dtype=np.float32), 24000)

        mock_backend.generate_speech = AsyncMock(side_effect=counting_generate)

        resp = client.post(
            "/v1/jobs",
            json={
                "chapters": [
                    {"text": "One."},
                    {"text": "Two."},
                    {"text": "Three."},
                    {"text": "Four."},
                ]
            },
        )
        job_id = resp.json()["id"]
        data = _poll_job(client, job_id)

        assert data["status"] == "completed"
        assert data["completed_chapters"] == 4
        first_run_calls = call_count

        # Verify state.json has all 4 chapters completed
        manager = get_job_manager()
        with open(manager._state_file) as f:
            state = json.load(f)
        assert all(
            ch["status"] == "completed" for ch in state[job_id]["chapters"]
        )

        # Simulate recovery: mark chapters 3 & 4 as pending (as if crash happened)
        state[job_id]["status"] = "processing"
        state[job_id]["chapters"][2]["status"] = "pending"
        state[job_id]["chapters"][2]["audio_filename"] = None
        state[job_id]["chapters"][3]["status"] = "pending"
        state[job_id]["chapters"][3]["audio_filename"] = None
        with open(manager._state_file, "w") as f:
            json.dump(state, f)

        # Delete chapter 3 & 4 audio files so checkpoint check sees them missing
        job_dir = manager._output_dir / job_id
        for p in job_dir.glob("chapter_0003*"):
            p.unlink()
        for p in job_dir.glob("chapter_0004*"):
            p.unlink()

        # Reset call counter
        call_count = 0

        # Create a fresh manager that recovers from state.json
        new_manager = JobManager(output_dir=str(manager._output_dir))
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(new_manager._recover_state())
        finally:
            loop.close()

        # Verify recovered job has chapters 1&2 completed, 3&4 pending
        recovered_job = new_manager._jobs[job_id]
        assert recovered_job.chapters[0].status == "completed"
        assert recovered_job.chapters[1].status == "completed"
        assert recovered_job.chapters[2].status == "pending"
        assert recovered_job.chapters[3].status == "pending"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestJobCancellation:

    def test_cancel_pending_job(self, client):
        """Cancel a job before it starts processing."""
        # Submit multiple jobs so the second one stays pending
        client.post(
            "/v1/jobs",
            json={
                "chapters": [{"text": f"Chapter {i}."} for i in range(20)]
            },
        )
        resp2 = client.post("/v1/jobs", json={"input": "Cancel me"})
        job_id = resp2.json()["id"]

        # Delete (which cancels first)
        del_resp = client.delete(f"/v1/jobs/{job_id}")
        assert del_resp.status_code == 204

    def test_cancel_sets_skipped_chapters(self, client, mock_backend):
        """Cancel mid-processing, verify remaining chapters are skipped."""
        call_count = 0

        async def slow_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return (np.zeros(2400, dtype=np.float32), 24000)

        mock_backend.generate_speech = AsyncMock(side_effect=slow_generate)

        resp = client.post(
            "/v1/jobs",
            json={
                "chapters": [{"text": f"Chapter {i}."} for i in range(10)]
            },
        )
        job_id = resp.json()["id"]

        # Wait briefly for processing to start, then cancel
        time.sleep(0.3)
        from api.services.job_manager import get_job_manager

        manager = get_job_manager()
        job = manager._jobs.get(job_id)
        if job and job.status in ("pending", "processing"):
            job.cancel_requested = True

        data = _poll_job(client, job_id, terminal_statuses={"completed", "completed_with_errors", "failed", "cancelled"})
        assert data["status"] == "cancelled"

        skipped = sum(1 for ch in data["chapters"] if ch["status"] == "skipped")
        assert skipped > 0


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------


class TestPriorityQueue:

    def test_interactive_request_works_during_batch(self, client, mock_backend):
        """Submit a batch job, then hit /v1/audio/speech — should still work."""
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.05)
            return (np.zeros(2400, dtype=np.float32), 24000)

        mock_backend.generate_speech = AsyncMock(side_effect=slow_generate)

        # Submit a large batch
        client.post(
            "/v1/jobs",
            json={
                "chapters": [{"text": f"Chapter {i}."} for i in range(5)]
            },
        )

        # Interactive request should still complete
        speech_resp = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Quick interactive request.", "voice": "Vivian"},
        )
        assert speech_resp.status_code == 200
        assert len(speech_resp.content) > 0


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------


class TestWebhookDelivery:

    def test_webhook_posted_on_completion(self, client, mock_backend):
        """Submit a job with webhook_url, verify delivery attempted."""
        with patch("api.services.job_manager.JobManager._fire_webhook") as mock_webhook:
            mock_webhook.return_value = None  # async mock

            resp = client.post(
                "/v1/jobs",
                json={
                    "input": "Webhook test.",
                    "webhook_url": "https://example.com/hook",
                },
            )
            job_id = resp.json()["id"]
            _poll_job(client, job_id)

            # The webhook should have been called
            assert mock_webhook.called

    def test_webhook_state_tracked(self, client, mock_backend):
        """Verify webhook_url and webhook_delivered appear in job response."""
        resp = client.post(
            "/v1/jobs",
            json={
                "input": "Webhook state test.",
                "webhook_url": "https://example.com/hook",
            },
        )
        job_id = resp.json()["id"]
        data = _poll_job(client, job_id)

        assert data["webhook_url"] == "https://example.com/hook"
        assert data["webhook_delivered"] is not None
