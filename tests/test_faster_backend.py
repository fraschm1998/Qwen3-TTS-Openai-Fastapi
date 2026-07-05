# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Behavioral tests for the faster-qwen3-tts backend.

These tests stub the `torch` and `faster_qwen3_tts` modules so they run
without a GPU or the optional dependency installed — matching how the rest
of the suite avoids loading real models.
"""

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.backends.faster_qwen3_tts import FasterQwen3TTSBackend


SAMPLE_RATE = 24000


def _fake_torch(cuda_available: bool) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.cuda = MagicMock()
    torch.cuda.is_available.return_value = cuda_available
    return torch


def _fake_faster_module(model: MagicMock) -> types.ModuleType:
    module = types.ModuleType("faster_qwen3_tts")

    class FasterQwen3TTS:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return model

    module.FasterQwen3TTS = FasterQwen3TTS
    return module


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.sample_rate = SAMPLE_RATE
    model.generate_custom_voice.return_value = (
        [np.ones(SAMPLE_RATE, dtype=np.float32)],
        SAMPLE_RATE,
    )
    model.generate_voice_clone.return_value = (
        [np.ones(SAMPLE_RATE // 2, dtype=np.float32)],
        SAMPLE_RATE,
    )
    return model


async def _initialized_backend(model=None, **backend_kwargs) -> FasterQwen3TTSBackend:
    model = model if model is not None else _fake_model()
    backend = FasterQwen3TTSBackend(**backend_kwargs)
    with patch.dict(
        sys.modules,
        {"torch": _fake_torch(True), "faster_qwen3_tts": _fake_faster_module(model)},
    ):
        await backend.initialize()
    return backend


class TestInitialization:
    async def test_requires_cuda(self):
        backend = FasterQwen3TTSBackend()
        with patch.dict(sys.modules, {"torch": _fake_torch(False)}):
            with pytest.raises(RuntimeError, match="requires an NVIDIA GPU"):
                await backend.initialize()
        assert not backend.is_ready()

    async def test_missing_package_gives_actionable_error(self):
        backend = FasterQwen3TTSBackend()
        with patch.dict(sys.modules, {"torch": _fake_torch(True), "faster_qwen3_tts": None}):
            with pytest.raises(RuntimeError, match="faster-qwen3-tts is not installed"):
                await backend.initialize()

    async def test_successful_initialize(self):
        backend = await _initialized_backend()
        assert backend.is_ready()
        assert backend.get_backend_name() == "faster"


class TestGeneration:
    async def test_generate_speech_returns_audio_and_sample_rate(self):
        model = _fake_model()
        backend = await _initialized_backend(model)

        audio, sr = await backend.generate_speech(
            text="Hello", voice="Vivian", language="English"
        )

        assert sr == SAMPLE_RATE
        assert len(audio) == SAMPLE_RATE
        kwargs = model.generate_custom_voice.call_args.kwargs
        assert kwargs["text"] == "Hello"
        assert kwargs["speaker"] == "Vivian"
        assert kwargs["language"] == "English"

    async def test_generation_is_serialized_by_lock(self):
        """Concurrent generations must not overlap (CUDA graph static buffers)."""
        model = _fake_model()
        active = 0
        max_active = 0

        def tracked_generate(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            import time
            time.sleep(0.02)
            active -= 1
            return [np.ones(10, dtype=np.float32)], SAMPLE_RATE

        model.generate_custom_voice.side_effect = tracked_generate
        backend = await _initialized_backend(model)

        await asyncio.gather(
            *(backend.generate_speech(text=f"t{i}", voice="Vivian") for i in range(4))
        )
        assert max_active == 1


class TestStreaming:
    async def test_streaming_yields_chunks_in_order(self):
        model = _fake_model()
        chunks = [np.full(10, i, dtype=np.float32) for i in range(3)]

        def fake_stream(**kwargs):
            for chunk in chunks:
                yield chunk, SAMPLE_RATE, {"steps": 1}

        model.generate_custom_voice_streaming.side_effect = fake_stream
        backend = await _initialized_backend(model)

        received = []
        async for chunk, sr in backend.generate_speech_streaming(
            text="Hello", voice="Vivian"
        ):
            assert sr == SAMPLE_RATE
            received.append(chunk)

        assert len(received) == 3
        for i, chunk in enumerate(received):
            assert np.all(chunk == i)
        assert not backend._generation_lock.locked()

    async def test_abandoned_stream_releases_lock(self):
        """Client disconnect must not deadlock subsequent requests."""
        model = _fake_model()

        def endless_stream(**kwargs):
            while True:
                yield np.zeros(10, dtype=np.float32), SAMPLE_RATE, {}

        model.generate_custom_voice_streaming.side_effect = endless_stream
        backend = await _initialized_backend(model)

        stream = backend.generate_speech_streaming(text="Hello", voice="Vivian")
        await stream.__anext__()
        assert backend._generation_lock.locked()

        await stream.aclose()
        assert not backend._generation_lock.locked()

        # A follow-up request must succeed.
        audio, sr = await backend.generate_speech(text="again", voice="Vivian")
        assert sr == SAMPLE_RATE

    async def test_mid_stream_error_raises_and_releases_lock(self):
        model = _fake_model()

        def failing_stream(**kwargs):
            yield np.zeros(10, dtype=np.float32), SAMPLE_RATE, {}
            raise RuntimeError("decode exploded")

        model.generate_custom_voice_streaming.side_effect = failing_stream
        backend = await _initialized_backend(model)

        with pytest.raises(RuntimeError, match="Streaming generation failed"):
            async for _chunk, _sr in backend.generate_speech_streaming(
                text="Hello", voice="Vivian"
            ):
                pass

        assert not backend._generation_lock.locked()


class TestVoiceClone:
    async def _clone_backend(self, model=None, tmp_path=None, monkeypatch=None):
        if monkeypatch and tmp_path:
            monkeypatch.setenv("TTS_REF_AUDIO_CACHE_DIR", str(tmp_path / "ref_cache"))
        return await _initialized_backend(
            model, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )

    async def test_clone_passes_cached_path_and_xvec_flag(self, tmp_path, monkeypatch):
        model = _fake_model()
        backend = await self._clone_backend(model, tmp_path, monkeypatch)
        ref = np.random.default_rng(0).uniform(-1, 1, SAMPLE_RATE).astype(np.float32)

        audio, sr = await backend.generate_voice_clone(
            text="Hello",
            ref_audio=ref,
            ref_audio_sr=SAMPLE_RATE,
            ref_text="reference transcript",
            x_vector_only_mode=True,
        )

        assert sr == SAMPLE_RATE
        kwargs = model.generate_voice_clone.call_args.kwargs
        assert kwargs["xvec_only"] is True
        assert kwargs["ref_text"] == "reference transcript"
        assert kwargs["ref_audio"].endswith(".wav")

    async def test_same_ref_audio_maps_to_same_path(self, tmp_path, monkeypatch):
        """Content-addressing lets the library's voice-prompt cache hit."""
        model = _fake_model()
        backend = await self._clone_backend(model, tmp_path, monkeypatch)
        ref = np.random.default_rng(1).uniform(-1, 1, SAMPLE_RATE).astype(np.float32)

        for _ in range(2):
            await backend.generate_voice_clone(
                text="Hello", ref_audio=ref, ref_audio_sr=SAMPLE_RATE,
                ref_text="t", x_vector_only_mode=True,
            )

        paths = {
            call.kwargs["ref_audio"]
            for call in model.generate_voice_clone.call_args_list
        }
        assert len(paths) == 1
        cache_dir = tmp_path / "ref_cache"
        assert len(list(cache_dir.glob("*.wav"))) == 1

    async def test_ref_cache_evicts_oldest(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TTS_REF_AUDIO_CACHE_MAX", "2")
        model = _fake_model()
        backend = await self._clone_backend(model, tmp_path, monkeypatch)

        rng = np.random.default_rng(2)
        for i in range(4):
            ref = rng.uniform(-1, 1, SAMPLE_RATE).astype(np.float32)
            await backend.generate_voice_clone(
                text="Hello", ref_audio=ref, ref_audio_sr=SAMPLE_RATE,
                ref_text="t", x_vector_only_mode=True,
            )

        cache_dir = tmp_path / "ref_cache"
        assert len(list(cache_dir.glob("*.wav"))) == 2

    async def test_oversized_ref_audio_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TTS_MAX_REF_AUDIO_SECONDS", "1")
        backend = await self._clone_backend(None, tmp_path, monkeypatch)
        ref = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)

        with pytest.raises(ValueError, match="maximum allowed"):
            await backend.generate_voice_clone(
                text="Hello", ref_audio=ref, ref_audio_sr=SAMPLE_RATE,
                ref_text="t", x_vector_only_mode=True,
            )

    async def test_empty_ref_audio_rejected(self, tmp_path, monkeypatch):
        backend = await self._clone_backend(None, tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="empty"):
            await backend.generate_voice_clone(
                text="Hello",
                ref_audio=np.zeros(0, dtype=np.float32),
                ref_audio_sr=SAMPLE_RATE,
                ref_text="t",
                x_vector_only_mode=True,
            )

    async def test_customvoice_model_rejects_cloning(self):
        backend = await _initialized_backend()  # default CustomVoice model

        with pytest.raises(RuntimeError, match="Voice cloning requires"):
            await backend.generate_voice_clone(
                text="Hello",
                ref_audio=np.zeros(SAMPLE_RATE, dtype=np.float32),
                ref_audio_sr=SAMPLE_RATE,
            )
