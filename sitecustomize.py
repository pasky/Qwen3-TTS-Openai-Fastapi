"""Project-wide runtime customizations.

Python automatically imports `sitecustomize` (if present on sys.path) during
startup. Because vLLM-Omni uses multiprocessing, this module is a convenient
place to apply monkey patches that must affect both the main API process and
worker subprocesses.

We only apply patches when TTS_BACKEND indicates vLLM-Omni.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def _is_vllm_omni_backend() -> bool:
    b = (os.getenv("TTS_BACKEND") or "").lower().strip()
    return b in {"vllm_omni", "vllm-omni", "vllm"}


if _is_vllm_omni_backend():
    try:
        from patches.vllm_omni_qwen3_tts import apply as _apply_vllm_omni_qwen3_tts

        _apply_vllm_omni_qwen3_tts()
        logger.info("Applied vLLM-Omni Qwen3-TTS runtime patches (sitecustomize)")
    except Exception as e:
        # Never fail startup due to patches.
        logger.warning("Failed to apply vLLM-Omni patches: %s", e)
