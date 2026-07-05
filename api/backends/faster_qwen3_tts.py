# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
faster-qwen3-tts backend implementation.

Wraps https://github.com/andimarafioti/faster-qwen3-tts, which captures the
talker + predictor decode steps as CUDA graphs (static KV cache) for a
5-10x throughput / TTFA improvement over the official implementation on
NVIDIA GPUs. Requires CUDA — there is deliberately no CPU fallback here;
use the official backend for CPU deployments.

Concurrency: CUDA graph replay uses static input/output buffers, so
concurrent generation calls on one model instance would corrupt each
other. All generation goes through a single asyncio.Lock. Note that a
streaming response holds the lock until the stream completes — on a
single-GPU deployment this is the same serialization the GPU imposes
anyway, but it means one long stream delays queued requests.
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, List, Dict, Any

import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

DEFAULT_VOICES = ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]
DEFAULT_LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", "German", "French",
    "Spanish", "Russian", "Portuguese", "Italian",
]

# Streaming granularity in codec steps (12 steps ≈ 1 second of audio).
# Smaller values reduce time-to-first-audio, larger values reduce decode
# overhead. Overridable via TTS_CHUNK_SIZE.
DEFAULT_CHUNK_SIZE = 12

# Reference-audio cache bounds (voice cloning). The library caches voice
# prompts keyed by file path, so identical reference audio must map to a
# stable path to get cache hits across requests.
DEFAULT_REF_CACHE_MAX_FILES = 32
DEFAULT_MAX_REF_AUDIO_SECONDS = 60.0

_STREAM_END = object()


class FasterQwen3TTSBackend(TTSBackend):
    """CUDA-graph accelerated Qwen3-TTS backend using faster-qwen3-tts."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the backend.

        Args:
            model_name: HuggingFace model identifier (Base, CustomVoice,
                or VoiceDesign variant; 0.6B or 1.7B)
        """
        super().__init__()
        self.model_name = model_name
        self._ready = False
        self._generation_lock = asyncio.Lock()
        self.chunk_size = self._read_chunk_size()
        self._ref_cache_dir = Path(
            os.getenv("TTS_REF_AUDIO_CACHE_DIR")
            or Path(tempfile.gettempdir()) / "qwen3_tts_ref_cache"
        )
        self._ref_cache_max_files = int(
            os.getenv("TTS_REF_AUDIO_CACHE_MAX", str(DEFAULT_REF_CACHE_MAX_FILES))
        )
        self._max_ref_audio_seconds = float(
            os.getenv("TTS_MAX_REF_AUDIO_SECONDS", str(DEFAULT_MAX_REF_AUDIO_SECONDS))
        )

    @staticmethod
    def _read_chunk_size() -> int:
        raw = os.getenv("TTS_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                f"Invalid TTS_CHUNK_SIZE={raw!r}, using default {DEFAULT_CHUNK_SIZE}"
            )
            return DEFAULT_CHUNK_SIZE
        if value < 1:
            logger.warning(
                f"TTS_CHUNK_SIZE must be >= 1, got {value}; using default {DEFAULT_CHUNK_SIZE}"
            )
            return DEFAULT_CHUNK_SIZE
        return value

    async def initialize(self) -> None:
        """Load the model and capture CUDA graphs (happens inside from_pretrained)."""
        if self._ready:
            logger.info("faster-qwen3-tts backend already initialized")
            return

        try:
            import torch
        except ImportError as e:
            raise RuntimeError(f"PyTorch is required for the faster backend: {e}")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The 'faster' backend requires an NVIDIA GPU (CUDA graphs). "
                "Use TTS_BACKEND=official for CPU deployments."
            )

        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as e:
            raise RuntimeError(
                "faster-qwen3-tts is not installed. Install it with "
                "'pip install .[faster]' (requires Python 3.10+ and torch>=2.5.1). "
                f"Import error: {e}"
            )

        self.device = "cuda"
        self.dtype = torch.bfloat16
        attn_implementation = os.getenv("TTS_ATTN_IMPLEMENTATION", "sdpa")

        logger.info(
            f"Loading Qwen3-TTS model '{self.model_name}' with CUDA graphs "
            f"(attn={attn_implementation})... graph capture runs at load time "
            f"and may take a while on first start."
        )

        try:
            # Model load + graph capture is blocking and heavy; keep the
            # event loop free during startup.
            self.model = await asyncio.to_thread(
                FasterQwen3TTS.from_pretrained,
                self.model_name,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
        except Exception as e:
            logger.error(f"Failed to load faster-qwen3-tts backend: {e}")
            raise RuntimeError(f"Failed to initialize faster-qwen3-tts backend: {e}")

        self._ready = True
        logger.info(
            f"faster-qwen3-tts backend loaded successfully "
            f"(sample_rate={self.model.sample_rate}, chunk_size={self.chunk_size})"
        )

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech using CUDA-graph accelerated custom-voice inference."""
        if not self._ready:
            await self.initialize()

        async with self._generation_lock:
            try:
                wavs, sr = await asyncio.to_thread(
                    self.model.generate_custom_voice,
                    text=text,
                    speaker=voice,
                    language=language,
                    instruct=instruct,
                )
            except Exception as e:
                logger.error(f"Speech generation failed: {e}")
                raise RuntimeError(f"Speech generation failed: {e}")

        audio = wavs[0]
        return await self._apply_speed(audio, speed), sr

    def supports_streaming(self) -> bool:
        """CUDA-graph streaming is the core capability of this backend."""
        return True

    async def generate_speech_streaming(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
    ) -> AsyncIterator[Tuple[np.ndarray, int]]:
        """Stream (audio_chunk, sample_rate) tuples as audio is decoded."""
        if not self._ready:
            await self.initialize()

        generator_factory = lambda: self.model.generate_custom_voice_streaming(
            text=text,
            speaker=voice,
            language=language,
            instruct=instruct,
            chunk_size=self.chunk_size,
        )
        # aclosing: async for does NOT close the inner generator on early
        # exit (client disconnect) — without it the generation lock stays
        # held until GC finalizes the generator.
        async with contextlib.aclosing(self._stream(generator_factory)) as stream:
            async for chunk, sr in stream:
                yield chunk, sr

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech by cloning a voice from reference audio."""
        if not self._ready:
            await self.initialize()

        self._require_voice_cloning()
        ref_path = await asyncio.to_thread(
            self._cache_ref_audio, ref_audio, ref_audio_sr
        )

        async with self._generation_lock:
            try:
                wavs, sr = await asyncio.to_thread(
                    self.model.generate_voice_clone,
                    text=text,
                    language=language,
                    ref_audio=str(ref_path),
                    ref_text=ref_text or "",
                    xvec_only=x_vector_only_mode,
                )
            except Exception as e:
                logger.error(f"Voice cloning failed: {e}")
                raise RuntimeError(f"Voice cloning failed: {e}")

        audio = wavs[0]
        return await self._apply_speed(audio, speed), sr

    async def generate_voice_clone_streaming(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
    ) -> AsyncIterator[Tuple[np.ndarray, int]]:
        """Stream voice-cloned (audio_chunk, sample_rate) tuples."""
        if not self._ready:
            await self.initialize()

        self._require_voice_cloning()
        ref_path = await asyncio.to_thread(
            self._cache_ref_audio, ref_audio, ref_audio_sr
        )

        generator_factory = lambda: self.model.generate_voice_clone_streaming(
            text=text,
            language=language,
            ref_audio=str(ref_path),
            ref_text=ref_text or "",
            xvec_only=x_vector_only_mode,
            chunk_size=self.chunk_size,
        )
        # aclosing: see generate_speech_streaming.
        async with contextlib.aclosing(self._stream(generator_factory)) as stream:
            async for chunk, sr in stream:
                yield chunk, sr

    async def _stream(self, generator_factory) -> AsyncIterator[Tuple[np.ndarray, int]]:
        """
        Bridge a blocking chunk generator into an async iterator.

        The generation lock is held for the whole stream and released even
        if the consumer disconnects mid-stream (generator .close() runs in
        the finally block, which also stops the underlying decode loop).
        """
        async with self._generation_lock:
            generator = generator_factory()
            try:
                while True:
                    item = await asyncio.to_thread(next, generator, _STREAM_END)
                    if item is _STREAM_END:
                        break
                    chunk, sr, _timing = item
                    yield chunk, sr
            except Exception as e:
                logger.error(f"Streaming generation failed mid-stream: {e}")
                raise RuntimeError(f"Streaming generation failed: {e}")
            finally:
                await asyncio.to_thread(generator.close)

    def _require_voice_cloning(self) -> None:
        if not self.supports_voice_cloning():
            raise RuntimeError(
                "Voice cloning requires a Base model (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base). "
                "The current model does not support voice cloning."
            )

    def _cache_ref_audio(self, ref_audio: np.ndarray, ref_audio_sr: int) -> Path:
        """
        Persist reference audio to a content-addressed WAV file.

        faster-qwen3-tts takes reference audio as a file path and caches
        the extracted voice prompt keyed by that path — a stable
        content-derived path makes repeated requests with the same
        reference voice hit that cache. The directory is app-owned (0700),
        writes are atomic, and the cache is bounded by evicting the
        least-recently-used files.
        """
        duration = len(ref_audio) / float(ref_audio_sr) if ref_audio_sr else 0.0
        if duration > self._max_ref_audio_seconds:
            raise ValueError(
                f"Reference audio is {duration:.1f}s; maximum allowed is "
                f"{self._max_ref_audio_seconds:.0f}s (set TTS_MAX_REF_AUDIO_SECONDS to change)"
            )
        if duration <= 0.0:
            raise ValueError("Reference audio is empty")

        import soundfile as sf

        self._ref_cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        audio_f32 = ref_audio.astype(np.float32)
        digest = hashlib.sha256(
            audio_f32.tobytes() + str(ref_audio_sr).encode()
        ).hexdigest()
        target = self._ref_cache_dir / f"{digest}.wav"

        if target.exists():
            target.touch()  # refresh mtime so LRU eviction keeps hot entries
            return target

        fd, tmp_path = tempfile.mkstemp(dir=self._ref_cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                sf.write(f, audio_f32, ref_audio_sr, format="WAV")
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._evict_ref_cache()
        return target

    def _evict_ref_cache(self) -> None:
        """Delete oldest cached reference files beyond the size bound."""
        try:
            entries = sorted(
                self._ref_cache_dir.glob("*.wav"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for stale in entries[self._ref_cache_max_files:]:
                stale.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Reference audio cache eviction failed: {e}")

    async def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        if speed == 1.0:
            return audio
        if not LIBROSA_AVAILABLE:
            logger.warning("Speed adjustment requested but librosa not available")
            return audio
        return await asyncio.to_thread(
            librosa.effects.time_stretch, audio.astype(np.float32), rate=speed
        )

    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "faster"

    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name

    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        if self._ready and self.model is not None:
            try:
                # FasterQwen3TTS wraps Qwen3TTSModel (.model), which wraps
                # the transformers model (.model.model).
                inner = self.model.model.model
                if hasattr(inner, "get_supported_speakers"):
                    speakers = inner.get_supported_speakers()
                    if speakers:
                        return list(speakers)
            except Exception as e:
                logger.warning(f"Could not get speakers from model: {e}")
        return list(DEFAULT_VOICES)

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        if self._ready and self.model is not None:
            try:
                inner = self.model.model.model
                if hasattr(inner, "get_supported_languages"):
                    languages = inner.get_supported_languages()
                    if languages:
                        return list(languages)
            except Exception as e:
                logger.warning(f"Could not get languages from model: {e}")
        return list(DEFAULT_LANGUAGES)

    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready

    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
        }

        try:
            import torch

            if torch.cuda.is_available():
                info["gpu_available"] = True
                device_idx = torch.cuda.current_device()
                info["gpu_name"] = torch.cuda.get_device_name(device_idx)

                props = torch.cuda.get_device_properties(device_idx)
                info["vram_total"] = f"{props.total_memory / 1024**3:.2f} GB"

                if self._ready:
                    allocated = torch.cuda.memory_allocated(device_idx)
                    info["vram_used"] = f"{allocated / 1024**3:.2f} GB"
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")

        return info

    def supports_voice_cloning(self) -> bool:
        """Voice cloning requires a Base model (not CustomVoice/VoiceDesign)."""
        return "Base" in self.model_name and "CustomVoice" not in self.model_name

    def get_model_type(self) -> str:
        """Return the model type (base, customvoice, or voicedesign)."""
        if "VoiceDesign" in self.model_name:
            return "voicedesign"
        if "Base" in self.model_name:
            return "base"
        if "CustomVoice" in self.model_name:
            return "customvoice"
        return "unknown"
