# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Pydantic schemas for the async job queue system.
"""

import enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChapterStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChapterInput(BaseModel):
    """A single chapter to synthesize."""

    title: Optional[str] = Field(None, description="Optional chapter title for labeling")
    text: str = Field(..., min_length=1, description="Text content to synthesize")


class JobCreateRequest(BaseModel):
    """Request to create a new TTS job. Accepts single text or array of chapters."""

    input: Optional[str] = Field(
        None,
        description="Single text input. Mutually exclusive with 'chapters'.",
    )
    chapters: Optional[list[ChapterInput]] = Field(
        None,
        description="Array of chapters for batch processing. Mutually exclusive with 'input'.",
    )
    model: str = Field(default="qwen3-tts")
    voice: str = Field(default="Vivian")
    language: Optional[str] = Field(default="Auto")
    instruct: Optional[str] = Field(default=None)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3"
    )
    webhook_url: Optional[str] = Field(
        None,
        description="URL to POST job status on completion/failure.",
    )

    @model_validator(mode="after")
    def validate_input_or_chapters(self) -> "JobCreateRequest":
        if self.input is not None and self.chapters is not None:
            raise ValueError("Provide either 'input' or 'chapters', not both.")
        if self.input is None and self.chapters is None:
            raise ValueError("Provide either 'input' or 'chapters'.")
        if self.input is not None:
            self.chapters = [ChapterInput(text=self.input)]
            self.input = None
        return self


class ChapterProgress(BaseModel):
    """Status of a single chapter within a job."""

    index: int
    title: Optional[str] = None
    status: ChapterStatus = ChapterStatus.PENDING
    error: Optional[str] = None
    audio_filename: Optional[str] = None
    duration_seconds: Optional[float] = None


class JobResponse(BaseModel):
    """Full job status response."""

    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    model: str
    voice: str
    language: Optional[str]
    response_format: str
    total_chapters: int
    completed_chapters: int
    current_chapter: Optional[int] = None
    chapters: list[ChapterProgress]
    webhook_url: Optional[str] = None
    webhook_delivered: Optional[bool] = None
    error: Optional[str] = None


class JobSummary(BaseModel):
    """Lightweight job listing without per-chapter detail."""

    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    total_chapters: int
    completed_chapters: int
    current_chapter: Optional[int] = None


class JobListResponse(BaseModel):
    """Response for GET /v1/jobs."""

    jobs: list[JobSummary]
    total: int


class WebhookPayload(BaseModel):
    """Payload sent to webhook_url on job completion or failure."""

    job_id: str
    status: JobStatus
    completed_at: Optional[str] = None
    total_chapters: int
    completed_chapters: int
    failed_chapters: int
    error: Optional[str] = None
