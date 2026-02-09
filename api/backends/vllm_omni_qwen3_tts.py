# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
vLLM-Omni Qwen3-TTS backend implementation.

This backend uses vLLM-Omni for faster inference with Qwen3-TTS models.
Uses the official vLLM-Omni API with correct imports from `vllm_omni`.

See: https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_tts/
"""

import os
import io
import logging
import asyncio
import pickle
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Set multiprocessing method for vLLM-Omni (must be done before import)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Optional librosa import for speed adjustment
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class VLLMOmniQwen3TTSBackend(TTSBackend):
    """vLLM-Omni backend for Qwen3-TTS.

    Uses the same input structure as the official vLLM-Omni example:
    - inputs["prompt"]
    - inputs["additional_information"] with list-wrapped fields
    - output.multimodal_output["audio"] and ["sr"]

    Notes on CUSTOM_VOICE:
    - For the Base model, Qwen3-TTS supports voice cloning. The "official" backend
      supports CUSTOM_VOICE by loading a pickled voice_clone_prompt and passing it
      to generate_voice_clone().
    - vLLM-Omni uses the same underlying model code (generate_voice_clone), so we
      can support CUSTOM_VOICE by passing voice_clone_prompt via additional_information.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        stage_configs_path: Optional[str] = None,
        enable_stats: bool = False,
        stage_init_timeout_s: int = 300,
        seed: int = 42,
        max_tokens: int = 2048,
    ):
        super().__init__()
        self.model_name = model_name
        # vLLM-Omni expects a YAML with top-level `stage_args:`.
        # Default to a local copy of the upstream qwen3_tts stage config with small perf tweaks.
        self.stage_configs_path = stage_configs_path or os.getenv(
            "VLLM_OMNI_STAGE_CONFIG",
            "config/qwen3_tts_fast.yaml",
        )
        self.enable_stats = enable_stats
        self.stage_init_timeout_s = stage_init_timeout_s
        self.seed = seed
        self.max_tokens = max_tokens

        self._ready = False
        self._lock = asyncio.Lock()
        self.omni = None
        self.sampling_params_list = None

        # CUSTOM_VOICE support (Base model only)
        self.custom_voice_path = os.getenv("CUSTOM_VOICE")
        self.custom_voice_prompt: Optional[List[Any]] = None

    def _load_custom_voice_prompt(self) -> Optional[List[Any]]:
        """Load pickled voice clone prompt items from CUSTOM_VOICE."""
        if not self.custom_voice_path:
            return None
        if self.custom_voice_prompt is not None:
            return self.custom_voice_prompt

        custom_voice_path = os.path.expanduser(self.custom_voice_path)
        if not os.path.isfile(custom_voice_path):
            raise RuntimeError(f"CUSTOM_VOICE file not found: {custom_voice_path}")

        try:
            with open(custom_voice_path, "rb") as handle:
                prompt_items = pickle.load(handle)
        except Exception as e:
            raise RuntimeError(f"Failed to load CUSTOM_VOICE prompt: {e}")

        # allow wrappers like {"items": [...]}
        if isinstance(prompt_items, dict) and "items" in prompt_items:
            prompt_items = prompt_items["items"]

        if not isinstance(prompt_items, list) or not prompt_items:
            raise RuntimeError("CUSTOM_VOICE prompt must be a non-empty list")

        # IMPORTANT: vLLM-Omni runs the model in a separate worker process.
        # Ensure any tensors inside the prompt are CPU tensors (GPU tensors can
        # break IPC). We keep the items as-is (list) but move tensors to CPU.
        try:
            import torch

            fixed: list[Any] = []
            for it in prompt_items:
                # Support both object-style items and dict-style items.
                if isinstance(it, dict):
                    ref_code = it.get("ref_code")
                    ref_spk = it.get("ref_spk_embedding")
                    xvec = bool(it.get("x_vector_only_mode", False))
                    icl = bool(it.get("icl_mode", not xvec))
                    ref_text = it.get("ref_text")
                    if isinstance(ref_code, torch.Tensor):
                        ref_code = ref_code.detach().to("cpu").contiguous()
                    if isinstance(ref_spk, torch.Tensor):
                        ref_spk = ref_spk.detach().to("cpu").contiguous()
                        # torch(bfloat16).numpy() is unsupported; vLLM-Omni serializes tensors via numpy().
                        if ref_spk.dtype == torch.bfloat16:
                            ref_spk = ref_spk.to(torch.float32)
                    fixed.append(
                        {
                            "ref_code": ref_code,
                            "ref_spk_embedding": ref_spk,
                            "x_vector_only_mode": xvec,
                            "icl_mode": icl,
                            "ref_text": ref_text,
                        }
                    )
                else:
                    ref_code = getattr(it, "ref_code", None)
                    ref_spk = getattr(it, "ref_spk_embedding", None)
                    xvec = bool(getattr(it, "x_vector_only_mode", False))
                    icl = bool(getattr(it, "icl_mode", not xvec))
                    ref_text = getattr(it, "ref_text", None)
                    if isinstance(ref_code, torch.Tensor):
                        ref_code = ref_code.detach().to("cpu").contiguous()
                    if isinstance(ref_spk, torch.Tensor):
                        ref_spk = ref_spk.detach().to("cpu").contiguous()
                        # torch(bfloat16).numpy() is unsupported; vLLM-Omni serializes tensors via numpy().
                        if ref_spk.dtype == torch.bfloat16:
                            ref_spk = ref_spk.to(torch.float32)
                    fixed.append(
                        {
                            "ref_code": ref_code,
                            "ref_spk_embedding": ref_spk,
                            "x_vector_only_mode": xvec,
                            "icl_mode": icl,
                            "ref_text": ref_text,
                        }
                    )

            prompt_items = fixed
        except Exception as e:
            logger.warning("Failed to normalize CUSTOM_VOICE prompt tensors to CPU: %s", e)

        self.custom_voice_prompt = prompt_items
        logger.info(f"Loaded CUSTOM_VOICE prompt from {custom_voice_path}")
        return self.custom_voice_prompt

    async def initialize(self) -> None:
        if self._ready:
            logger.info("vLLM-Omni backend already initialized")
            return

        async with self._lock:
            if self._ready:
                return

            try:
                from vllm import SamplingParams
                from vllm_omni import Omni

                logger.info(f"Loading vLLM-Omni model '{self.model_name}'...")

                self.omni = Omni(
                    model=self.model_name,
                    stage_configs_path=self.stage_configs_path,
                    log_stats=self.enable_stats,
                    stage_init_timeout=self.stage_init_timeout_s,
                )

                # Pre-create sampling params for reuse
                self.sampling_params_list = [
                    SamplingParams(
                        temperature=0.9,
                        top_p=1.0,
                        top_k=50,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                        detokenize=False,
                        repetition_penalty=1.05,
                    )
                ]

                # warm-load custom prompt if requested
                if self.custom_voice_path and self.supports_voice_cloning():
                    self._load_custom_voice_prompt()

                self._ready = True
                logger.info("vLLM-Omni backend loaded successfully!")

            except ImportError as e:
                logger.error(f"vLLM-Omni not installed: {e}")
                raise RuntimeError(
                    "vLLM-Omni is not installed. Please use Dockerfile.vllm or install: "
                    "pip install vllm-omni (requires Python 3.12 and CUDA)"
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
        if not self._ready:
            await self.initialize()

        try:
            voice_key = (voice or "").lower()

            # Build prompt and inputs following official vLLM-Omni example
            prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

            # Determine task type based on model (controls which codepath in Qwen3-TTS is used)
            if "CustomVoice" in self.model_name:
                task_type = "CustomVoice"
            elif "VoiceDesign" in self.model_name:
                task_type = "VoiceDesign"
            else:
                task_type = "Base"

            additional_information: Dict[str, Any] = {
                "task_type": [task_type],
                "text": [text],
                "instruct": [instruct or ""],
                "language": [language],
                "speaker": [voice],
                "max_new_tokens": [self.max_tokens],
            }

            # If CUSTOM_VOICE is requested, pass voice_clone_prompt.
            # This is only meaningful for Base model.
            if voice_key == "custom":
                if not self.supports_voice_cloning():
                    raise RuntimeError(
                        "CUSTOM_VOICE requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                        "Set TTS_MODEL_NAME accordingly."
                    )
                prompt_items = self._load_custom_voice_prompt() or []

                # NOTE: vLLM v1 serializes requests via msgpack with limited type
                # information for `list[Any]`. Tensors nested inside lists/dicts can
                # arrive in the worker as plain Python lists (no `.to()`), breaking
                # the Qwen3-TTS clone path.
                #
                # Workaround: send the precomputed tensors as *top-level* additional_information
                # tensor entries (not inside list_data) and let the worker wrapper
                # reconstruct `voice_clone_prompt`.
                if not prompt_items:
                    raise RuntimeError("CUSTOM_VOICE prompt is empty")
                it0 = prompt_items[0]
                ref_code = it0.get("ref_code")
                ref_spk = it0.get("ref_spk_embedding")
                xvec = bool(it0.get("x_vector_only_mode", False))
                icl = bool(it0.get("icl_mode", not xvec))
                ref_text0 = it0.get("ref_text")

                if ref_spk is None:
                    raise RuntimeError("CUSTOM_VOICE prompt missing ref_spk_embedding")

                # Safety: if ref_code is missing, force x-vector mode.
                if ref_code is None:
                    xvec = True
                    icl = False

                additional_information["voice_clone_ref_spk_embedding"] = ref_spk
                if ref_code is not None:
                    additional_information["voice_clone_ref_code"] = ref_code
                additional_information["voice_clone_x_vector_only_mode"] = [xvec]
                additional_information["voice_clone_icl_mode"] = [icl]
                if ref_text0:
                    additional_information["voice_clone_ref_text"] = [str(ref_text0)]

                # Force Base/voice-clone path.
                additional_information["task_type"] = ["Base"]
                # Speaker is unused in Base/voice-clone path, but avoid sending "custom".
                additional_information["speaker"] = ["uncle_fu"]

            inputs = {
                "prompt": prompt,
                "additional_information": additional_information,
            }

            omni_generator = self.omni.generate(inputs, self.sampling_params_list)

            for stage_outputs in omni_generator:
                for output in stage_outputs.request_output:
                    audio_tensor = output.multimodal_output["audio"]
                    sr = int(output.multimodal_output["sr"].item())

                    audio_np = audio_tensor.float().detach().cpu().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()

                    if speed != 1.0 and LIBROSA_AVAILABLE:
                        audio_np = librosa.effects.time_stretch(audio_np.astype(np.float32), rate=speed)
                    elif speed != 1.0:
                        logger.warning("Speed adjustment requested but librosa not available")

                    return audio_np, sr

            raise RuntimeError("No audio returned from vLLM-Omni (no stage outputs)")

        except Exception as e:
            logger.error(f"vLLM-Omni speech generation failed: {e}")
            raise RuntimeError(f"vLLM-Omni speech generation failed: {e}")

    def synthesize_wav_bytes(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Auto",
        instruct: str = "",
        task_type: str = "CustomVoice",
    ) -> Tuple[bytes, int]:
        import soundfile as sf

        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.generate_speech(text, speaker, language, instruct))
                audio_np, sr = future.result()
        else:
            audio_np, sr = loop.run_until_complete(self.generate_speech(text, speaker, language, instruct))

        buf = io.BytesIO()
        sf.write(buf, audio_np, samplerate=sr, format="WAV")
        return buf.getvalue(), sr

    def close(self):
        if self.omni is not None:
            try:
                self.omni.close()
            except Exception as e:
                logger.warning(f"Error closing vLLM-Omni: {e}")
            finally:
                self.omni = None
                self._ready = False

    def get_backend_name(self) -> str:
        return "vllm_omni"

    def get_model_id(self) -> str:
        return self.model_name

    def get_supported_voices(self) -> List[str]:
        return [
            "Vivian",
            "Ryan",
            "Sophia",
            "Isabella",
            "Evan",
            "Lily",
            "Serena",
            "Dylan",
            "Eric",
            "Aiden",
        ]

    def get_supported_languages(self) -> List[str]:
        return [
            "Auto",
            "English",
            "Chinese",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Spanish",
            "Russian",
            "Portuguese",
            "Italian",
        ]

    def is_ready(self) -> bool:
        return self._ready

    def get_device_info(self) -> Dict[str, Any]:
        info = {
            "device": "cuda",
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
        return "Base" in self.model_name and "CustomVoice" not in self.model_name

    def get_model_type(self) -> str:
        if "Base" in self.model_name:
            return "base"
        if "CustomVoice" in self.model_name:
            return "customvoice"
        if "VoiceDesign" in self.model_name:
            return "voicedesign"
        return "unknown"
