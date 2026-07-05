# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
TTS engine service.

Core speech generation logic used by both the OpenAI-compatible router
and the async job system.
"""

import contextlib
import logging
from typing import AsyncIterator, Callable, Optional

import numpy as np

from .text_processing import split_into_chunks

logger = logging.getLogger(__name__)

# OpenAI voice mapping to Qwen voices
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Sophia",
    "nova": "Isabella",
    "onyx": "Evan",
    "shimmer": "Lily",
}

# Silence gap between chunks: 0.3 seconds
SILENCE_DURATION_SECONDS = 0.3


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed."""
    from ..backends import get_backend, initialize_backend

    backend = get_backend()

    if not backend.is_ready():
        await initialize_backend()

    return backend


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text using the configured TTS backend.

    Args:
        text: The text to synthesize
        voice: Voice name to use
        language: Language code
        instruct: Optional instruction for voice style
        speed: Speech speed multiplier

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    backend = await get_tts_backend()

    voice_name = get_voice_name(voice)

    try:
        audio, sr = await backend.generate_speech(
            text=text,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )

        return audio, sr

    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


class CancelledError(Exception):
    """Raised when a generation is cancelled via cancel_check callback."""


async def generate_speech_chunked(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text, splitting long text into chunks.

    Short text passes through to generate_speech() with no overhead.
    Long text is split at sentence boundaries, each chunk is synthesized
    sequentially, and the resulting audio arrays are concatenated with
    brief silence gaps.

    Args:
        cancel_check: Optional callable returning True if generation should
            be aborted. Checked between every chunk. Raises CancelledError.
    """
    chunks = split_into_chunks(text)

    if len(chunks) <= 1:
        return await generate_speech(
            text=text,
            voice=voice,
            language=language,
            instruct=instruct,
            speed=speed,
        )

    logger.info(f"Splitting text into {len(chunks)} chunks for synthesis")

    audio_segments: list[np.ndarray] = []
    sample_rate: int | None = None

    for i, chunk in enumerate(chunks):
        # Check cancellation between chunks
        if cancel_check and cancel_check():
            raise CancelledError("Generation cancelled")

        logger.debug(f"Generating chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")

        chunk_audio, chunk_sr = await generate_speech(
            text=chunk,
            voice=voice,
            language=language,
            instruct=instruct,
            speed=speed,
        )

        if sample_rate is None:
            sample_rate = chunk_sr

        audio_segments.append(chunk_audio)

        # Add silence gap between chunks (not after the last one)
        if i < len(chunks) - 1:
            silence_samples = int(chunk_sr * SILENCE_DURATION_SECONDS)
            silence = np.zeros(silence_samples, dtype=np.float32)
            audio_segments.append(silence)

    combined_audio = np.concatenate(audio_segments)
    return combined_audio, sample_rate


async def generate_speech_streaming(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
) -> AsyncIterator[tuple[np.ndarray, int]]:
    """
    Incrementally generate speech, yielding (audio_chunk, sample_rate)
    tuples as the backend decodes audio.

    Long text is split at sentence boundaries and streamed chunk-by-chunk
    sequentially, with the same silence gaps generate_speech_chunked()
    inserts. Requires a backend where supports_streaming() is True.

    Note: this path intentionally bypasses the JobManager priority queue —
    the backend serializes GPU access itself, and a live stream cannot be
    queued without destroying its latency benefit.
    """
    backend = await get_tts_backend()
    voice_name = get_voice_name(voice)
    text_chunks = split_into_chunks(text)

    for i, text_chunk in enumerate(text_chunks):
        chunk_sr: int | None = None

        # aclosing ensures the backend generator (and its GPU lock) is
        # released promptly if the consumer disconnects mid-stream.
        async with contextlib.aclosing(
            backend.generate_speech_streaming(
                text=text_chunk,
                voice=voice_name,
                language=language,
                instruct=instruct,
            )
        ) as stream:
            async for audio_chunk, sr in stream:
                chunk_sr = sr
                yield audio_chunk, sr

        # Silence gap between text chunks (not after the last one)
        if i < len(text_chunks) - 1 and chunk_sr:
            silence = np.zeros(
                int(chunk_sr * SILENCE_DURATION_SECONDS), dtype=np.float32
            )
            yield silence, chunk_sr
