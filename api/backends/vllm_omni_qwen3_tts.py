# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
vLLM-Omni Qwen3-TTS backend implementation.

This backend uses vLLM-Omni for faster inference with Qwen3-TTS models.
Note: vLLM-Omni currently only supports offline inference.
"""

import logging
import asyncio
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class VLLMOmniQwen3TTSBackend(TTSBackend):
    """vLLM-Omni backend for Qwen3-TTS."""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        max_model_len: int = 2048,
    ):
        """
        Initialize the vLLM-Omni backend.
        
        Args:
            model_name: HuggingFace model identifier (recommend 0.6B for speed)
            max_model_len: Maximum model length for vLLM
        """
        super().__init__()
        self.model_name = model_name
        self.max_model_len = max_model_len
        self._ready = False
        self._lock = asyncio.Lock()  # For thread safety
        self.llm = None
    
    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("vLLM-Omni backend already initialized")
            return
        
        async with self._lock:
            if self._ready:
                return
            
            try:
                import torch
                from vllm import Omni
                
                # Determine device
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                    self.dtype = torch.bfloat16
                else:
                    self.device = "cpu"
                    self.dtype = torch.float32
                
                logger.info(f"Loading vLLM-Omni model '{self.model_name}' on {self.device}...")
                
                # Initialize vLLM-Omni
                self.llm = Omni(
                    model=self.model_name,
                    max_model_len=self.max_model_len,
                    dtype="bfloat16" if self.dtype == torch.bfloat16 else "float32",
                )
                
                self._ready = True
                logger.info(f"vLLM-Omni backend loaded successfully on {self.device}")
                
            except ImportError as e:
                logger.error(f"vLLM-Omni not installed: {e}")
                raise RuntimeError(
                    "vLLM-Omni is not installed. Please install with: pip install vllm"
                )
            except Exception as e:
                logger.error(f"Failed to load vLLM-Omni backend: {e}")
                raise RuntimeError(f"Failed to initialize vLLM-Omni backend: {e}")
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text using vLLM-Omni.
        
        Args:
            text: The text to synthesize
            voice: Voice name to use (speaker id)
            language: Language code
            instruct: Optional instruction for voice style
            speed: Speech speed multiplier
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()
        
        # Use lock to ensure thread safety for vLLM-Omni
        async with self._lock:
            try:
                from vllm import SamplingParams
                
                # Prepare the prompt for CustomVoice task
                # Format based on vLLM-Omni's offline example
                prompt = self._build_prompt(text, voice, language, instruct)
                
                # Set up sampling parameters
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=self.max_model_len,
                )
                
                # Generate using vLLM-Omni
                outputs = self.llm.generate(
                    prompts=[prompt],
                    sampling_params=sampling_params,
                )
                
                # Extract audio from output
                # Note: This is a simplified version - actual implementation 
                # depends on vLLM-Omni's output format
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    
                    # Get audio data from output
                    # vLLM-Omni should return audio in the output
                    audio_data = self._extract_audio_from_output(output)
                    
                    # Default sample rate for Qwen3-TTS-12Hz models
                    sample_rate = 12000
                    
                    # Apply speed adjustment if needed
                    if speed != 1.0 and LIBROSA_AVAILABLE:
                        audio_data = librosa.effects.time_stretch(
                            audio_data.astype(np.float32), 
                            rate=speed
                        )
                    elif speed != 1.0:
                        logger.warning("Speed adjustment requested but librosa not available")
                    
                    return audio_data, sample_rate
                else:
                    raise RuntimeError("No output generated from vLLM-Omni")
                
            except Exception as e:
                logger.error(f"vLLM-Omni speech generation failed: {e}")
                raise RuntimeError(f"vLLM-Omni speech generation failed: {e}")
    
    def _build_prompt(
        self, 
        text: str, 
        voice: str, 
        language: str, 
        instruct: Optional[str]
    ) -> str:
        """
        Build the prompt for vLLM-Omni CustomVoice task.
        
        Args:
            text: Text to synthesize
            voice: Speaker name
            language: Language
            instruct: Optional instruction
        
        Returns:
            Formatted prompt string
        """
        # This format should match vLLM-Omni's expected input
        # Adjust based on actual vLLM-Omni API
        parts = [f"<|text|>{text}<|/text|>"]
        
        if voice:
            parts.append(f"<|speaker|>{voice}<|/speaker|>")
        
        if language and language != "Auto":
            parts.append(f"<|language|>{language}<|/language|>")
        
        if instruct:
            parts.append(f"<|instruct|>{instruct}<|/instruct|>")
        
        return "".join(parts)
    
    def _extract_audio_from_output(self, output) -> np.ndarray:
        """
        Extract audio data from vLLM-Omni output.
        
        Args:
            output: vLLM-Omni output object
        
        Returns:
            Audio as numpy array
        """
        # This is a placeholder - actual implementation depends on 
        # vLLM-Omni's output format
        # The output should contain audio data that we can extract
        
        if hasattr(output, 'audio'):
            return np.array(output.audio, dtype=np.float32)
        elif hasattr(output, 'outputs') and len(output.outputs) > 0:
            # Try to get audio from outputs
            output_data = output.outputs[0]
            if hasattr(output_data, 'audio'):
                return np.array(output_data.audio, dtype=np.float32)
        
        # Fallback: try to convert output text to audio tokens
        # This would need the tokenizer/decoder
        logger.warning("Could not extract audio from vLLM-Omni output directly")
        raise RuntimeError("Failed to extract audio from vLLM-Omni output")
    
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "vllm_omni"
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name
    
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        # vLLM-Omni with Qwen3-TTS supports the same voices as official
        return ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        # vLLM-Omni with Qwen3-TTS supports the same languages as official
        return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                "Spanish", "Russian", "Portuguese", "Italian"]
    
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
                if torch.cuda.current_device() >= 0:
                    device_idx = torch.cuda.current_device()
                    info["gpu_name"] = torch.cuda.get_device_name(device_idx)
                    
                    # Get VRAM info
                    props = torch.cuda.get_device_properties(device_idx)
                    info["vram_total"] = f"{props.total_memory / 1024**3:.2f} GB"
                    
                    if self._ready:
                        allocated = torch.cuda.memory_allocated(device_idx)
                        info["vram_used"] = f"{allocated / 1024**3:.2f} GB"
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")
        
        return info
