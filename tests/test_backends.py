# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for backend selection and initialization.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.backends.factory import get_backend, reset_backend
from api.backends.base import TTSBackend
from api.backends.faster_qwen3_tts import FasterQwen3TTSBackend
from api.backends.official_qwen3_tts import OfficialQwen3TTSBackend
from api.backends.vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend


class TestBackendSelection:
    """Test backend selection via environment variables."""
    
    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()
    
    def test_default_backend_is_official(self, monkeypatch):
        """Test that official backend is selected by default."""
        # Ensure TTS_BACKEND is not set
        monkeypatch.delenv("TTS_BACKEND", raising=False)
        
        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"
    
    def test_official_backend_via_env(self, monkeypatch):
        """Test selecting official backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        
        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"
    
    def test_vllm_backend_via_env(self, monkeypatch):
        """Test selecting vLLM-Omni backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "vllm_omni")
        
        backend = get_backend()
        assert isinstance(backend, VLLMOmniQwen3TTSBackend)
        assert backend.get_backend_name() == "vllm_omni"
    
    def test_vllm_backend_alternate_name(self, monkeypatch):
        """Test vLLM backend with alternate name format."""
        monkeypatch.setenv("TTS_BACKEND", "vllm-omni")
        
        backend = get_backend()
        assert isinstance(backend, VLLMOmniQwen3TTSBackend)
        assert backend.get_backend_name() == "vllm_omni"
    
    def test_faster_backend_via_env(self, monkeypatch):
        """Test selecting faster-qwen3-tts backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "faster")

        backend = get_backend()
        assert isinstance(backend, FasterQwen3TTSBackend)
        assert backend.get_backend_name() == "faster"

    def test_faster_backend_alternate_names(self, monkeypatch):
        """Test faster backend with alternate name formats."""
        for name in ("faster-qwen3-tts", "faster_qwen3_tts"):
            reset_backend()
            monkeypatch.setenv("TTS_BACKEND", name)

            backend = get_backend()
            assert isinstance(backend, FasterQwen3TTSBackend)

    def test_faster_backend_custom_model(self, monkeypatch):
        """Test overriding model name for the faster backend."""
        monkeypatch.setenv("TTS_BACKEND", "faster")
        monkeypatch.setenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

        backend = get_backend()
        assert backend.get_model_id() == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    def test_invalid_backend_raises_error(self, monkeypatch):
        """Test that invalid backend name raises ValueError."""
        monkeypatch.setenv("TTS_BACKEND", "invalid_backend")
        
        with pytest.raises(ValueError, match="Unknown TTS_BACKEND"):
            get_backend()
    
    def test_custom_model_name_via_env(self, monkeypatch):
        """Test overriding model name via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        monkeypatch.setenv("TTS_MODEL_NAME", "custom/model")
        
        backend = get_backend()
        assert backend.get_model_id() == "custom/model"
    
    def test_backend_singleton(self, monkeypatch):
        """Test that get_backend returns the same instance."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        
        backend1 = get_backend()
        backend2 = get_backend()
        
        assert backend1 is backend2


class TestBackendInterface:
    """Test that all backends implement the required interface."""
    
    def test_official_backend_implements_interface(self):
        """Test official backend implements TTSBackend interface."""
        backend = OfficialQwen3TTSBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
    
    def test_vllm_backend_implements_interface(self):
        """Test vLLM backend implements TTSBackend interface."""
        backend = VLLMOmniQwen3TTSBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
    
    def test_backend_names_are_correct(self):
        """Test that backends return correct names."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert official.get_backend_name() == "official"
        assert vllm.get_backend_name() == "vllm_omni"
    
    def test_backends_return_voices(self):
        """Test that backends return voice lists."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        # Both backends should return a list of voices
        assert isinstance(official.get_supported_voices(), list)
        assert isinstance(vllm.get_supported_voices(), list)
        assert len(official.get_supported_voices()) > 0
        assert len(vllm.get_supported_voices()) > 0
    
    def test_backends_return_languages(self):
        """Test that backends return language lists."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        # Both backends should return a list of languages
        assert isinstance(official.get_supported_languages(), list)
        assert isinstance(vllm.get_supported_languages(), list)
        assert len(official.get_supported_languages()) > 0
        assert len(vllm.get_supported_languages()) > 0
    
    def test_backends_initially_not_ready(self):
        """Test that backends are not ready before initialization."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert not official.is_ready()
        assert not vllm.is_ready()
    
    def test_backends_return_device_info(self):
        """Test that backends return device info dict."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        info1 = official.get_device_info()
        info2 = vllm.get_device_info()
        
        # Check required keys
        assert "device" in info1
        assert "gpu_available" in info1
        assert "device" in info2
        assert "gpu_available" in info2


class TestFasterBackendInterface:
    """Tests for the faster-qwen3-tts backend interface."""

    def test_implements_interface(self):
        """Test faster backend implements TTSBackend interface."""
        backend = FasterQwen3TTSBackend()

        assert isinstance(backend, TTSBackend)
        for method in (
            'initialize', 'generate_speech', 'get_backend_name', 'get_model_id',
            'get_supported_voices', 'get_supported_languages', 'is_ready',
            'get_device_info', 'supports_voice_cloning', 'get_model_type',
            'generate_voice_clone', 'supports_streaming',
            'generate_speech_streaming', 'generate_voice_clone_streaming',
        ):
            assert hasattr(backend, method)

    def test_initially_not_ready(self):
        backend = FasterQwen3TTSBackend()
        assert not backend.is_ready()

    def test_returns_default_voices_and_languages_when_not_loaded(self):
        backend = FasterQwen3TTSBackend()
        assert len(backend.get_supported_voices()) > 0
        assert len(backend.get_supported_languages()) > 0

    def test_streaming_capability_flags(self):
        """Only the faster backend advertises streaming."""
        assert FasterQwen3TTSBackend().supports_streaming()
        assert not OfficialQwen3TTSBackend().supports_streaming()
        assert not VLLMOmniQwen3TTSBackend().supports_streaming()

    def test_cloning_capability_matrix(self):
        """Voice cloning follows the same model-name rules as other backends."""
        base = FasterQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        custom = FasterQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        design = FasterQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

        assert base.supports_voice_cloning()
        assert base.get_model_type() == "base"
        assert not custom.supports_voice_cloning()
        assert custom.get_model_type() == "customvoice"
        assert not design.supports_voice_cloning()
        assert design.get_model_type() == "voicedesign"

    def test_chunk_size_from_env(self, monkeypatch):
        monkeypatch.setenv("TTS_CHUNK_SIZE", "4")
        assert FasterQwen3TTSBackend().chunk_size == 4

    def test_invalid_chunk_size_falls_back_to_default(self, monkeypatch):
        from api.backends.faster_qwen3_tts import DEFAULT_CHUNK_SIZE

        monkeypatch.setenv("TTS_CHUNK_SIZE", "not-a-number")
        assert FasterQwen3TTSBackend().chunk_size == DEFAULT_CHUNK_SIZE

        monkeypatch.setenv("TTS_CHUNK_SIZE", "0")
        assert FasterQwen3TTSBackend().chunk_size == DEFAULT_CHUNK_SIZE

    def test_streaming_not_supported_on_other_backends(self):
        """Base-class default raises NotImplementedError."""
        backend = OfficialQwen3TTSBackend()
        with pytest.raises(NotImplementedError):
            backend.generate_speech_streaming(text="hi", voice="Vivian")


class TestVoiceCloningInterface:
    """Tests for voice cloning interface across all backends."""

    def test_official_backend_has_voice_cloning_methods(self):
        """Test that official backend has voice cloning methods."""
        backend = OfficialQwen3TTSBackend()
        
        assert hasattr(backend, 'supports_voice_cloning')
        assert hasattr(backend, 'get_model_type')
        assert hasattr(backend, 'generate_voice_clone')

    def test_vllm_backend_has_voice_cloning_methods(self):
        """Test that vLLM backend has voice cloning methods."""
        backend = VLLMOmniQwen3TTSBackend()
        
        assert hasattr(backend, 'supports_voice_cloning')
        assert hasattr(backend, 'get_model_type')

    def test_customvoice_model_does_not_support_cloning(self):
        """Test that CustomVoice models don't support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        assert not official.supports_voice_cloning()
        assert not vllm.supports_voice_cloning()
        assert official.get_model_type() == "customvoice"
        assert vllm.get_model_type() == "customvoice"

    def test_base_model_supports_cloning(self):
        """Test that Base models support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        
        assert official.supports_voice_cloning()
        assert vllm.supports_voice_cloning()
        assert official.get_model_type() == "base"
        assert vllm.get_model_type() == "base"

    def test_voicedesign_model_does_not_support_cloning(self):
        """Test that VoiceDesign models don't support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        
        assert not official.supports_voice_cloning()
        assert not vllm.supports_voice_cloning()

    def test_vllm_backend_voicedesign_model_type(self):
        """Test vLLM backend returns correct model type for VoiceDesign."""
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        
        assert vllm.get_model_type() == "voicedesign"

    def test_model_type_defaults_to_customvoice(self):
        """Test that default model type is customvoice."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert official.get_model_type() == "customvoice"
        assert vllm.get_model_type() == "customvoice"
