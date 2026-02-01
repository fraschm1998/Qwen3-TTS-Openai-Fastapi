# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Router for async TTS job management.
"""

import logging

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from ..services.audio_encoding import get_content_type
from ..services.job_manager import get_job_manager
from ..structures.job_schemas import (
    JobCreateRequest,
    JobListResponse,
    JobResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Job Queue"],
    responses={404: {"description": "Not found"}},
)


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(request: JobCreateRequest):
    """
    Create a new TTS job.

    Accepts either a single ``input`` text or an array of ``chapters``
    for batch processing. Returns immediately with a job ID that can
    be polled for progress.
    """
    manager = get_job_manager()
    return await manager.submit_job(request)


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all TTS jobs with summary status."""
    manager = get_job_manager()
    return manager.list_jobs()


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get full status and chapter-level progress for a job."""
    manager = get_job_manager()
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job '{job_id}' not found"},
        )
    return job


@router.get("/jobs/{job_id}/audio/{chapter}")
async def download_chapter_audio(job_id: str, chapter: int):
    """
    Download the audio file for a specific chapter (1-indexed).
    """
    manager = get_job_manager()

    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job '{job_id}' not found"},
        )

    if chapter < 1 or chapter > job.total_chapters:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "invalid_chapter",
                "message": f"Chapter {chapter} out of range (1-{job.total_chapters})",
            },
        )

    path = manager.get_chapter_audio_path(job_id, chapter)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "audio_not_ready",
                "message": f"Audio for chapter {chapter} is not available yet",
            },
        )

    ext = path.suffix.lstrip(".")
    media_type = get_content_type(ext)
    return FileResponse(path=str(path), media_type=media_type, filename=path.name)


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Cancel and delete a job and its audio files."""
    manager = get_job_manager()
    deleted = await manager.delete_job(job_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job '{job_id}' not found"},
        )
    return Response(status_code=204)
