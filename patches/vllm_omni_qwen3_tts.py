"""Patches for vllm_omni's Qwen3-TTS integration.

Why patches are needed
----------------------

vLLM v1 uses msgpack serialization between the API process and worker
processes. While vllm_omni's `additional_information` supports tensor entries,
*nested* torch tensors inside list/dict payloads are not reliably reconstructed
as torch.Tensor objects on the worker side.

For CUSTOM_VOICE, this repo sends precomputed voice-clone conditioning tensors
(speaker embedding and ref codes) via top-level additional_information keys.
These patches make vllm_omni's Qwen3-TTS wrapper consume those fields without
mutating per-request state.

Patch summary
-------------

1) Wrap Qwen3TTSModelForGeneration.forward() to avoid mutating the shared
   per-request `runtime_additional_information` dict across iterations.
2) Wrap Qwen3TTSModel.generate_voice_clone() so it can accept the
   server-provided helper keys:
      - voice_clone_ref_spk_embedding (Tensor)
      - voice_clone_ref_code (Tensor, optional)
      - voice_clone_x_vector_only_mode (bool)
      - voice_clone_icl_mode (bool)
      - voice_clone_ref_text (str)
   and convert them into a proper VoiceClonePromptItem list.

The goal is to support `voice=custom` with vLLM-Omni for the Base model.

These patches are best-effort and will no-op if vllm_omni is not installed.
"""

from __future__ import annotations

from typing import Any, Callable


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, list) and x:
        return bool(x[0])
    return bool(x)


def _as_str(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, list) and x:
        x = x[0]
    return str(x)


def apply() -> None:
    """Apply monkey patches in-process (safe to call multiple times)."""
    try:
        from vllm_omni.model_executor.models.qwen3_tts import qwen3_tts as q3
    except Exception:
        return

    # ---------------------------------------------------------------------
    # 1) forward(): avoid mutating shared per-request dict (pop() etc.)
    # ---------------------------------------------------------------------
    try:
        cls = q3.Qwen3TTSModelForGeneration
        if getattr(cls, "__patched_no_mutation__", False) is False:
            orig_forward: Callable[..., Any] = cls.forward

            def forward_no_mutation(self, *args: Any, **kwargs: Any):
                rai = kwargs.get("runtime_additional_information")
                if isinstance(rai, list) and rai and isinstance(rai[0], dict):
                    # Shallow copy; values inside are (mostly) lists/tensors.
                    kwargs = dict(kwargs)
                    kwargs["runtime_additional_information"] = [dict(rai[0])]
                return orig_forward(self, *args, **kwargs)

            cls.forward = forward_no_mutation  # type: ignore[assignment]
            cls.__patched_no_mutation__ = True
    except Exception:
        # Don't break import if internals changed.
        pass

    # ---------------------------------------------------------------------
    # 2) generate_voice_clone(): accept helper keys from server
    # ---------------------------------------------------------------------
    try:
        model_cls = q3.Qwen3TTSModel
        if getattr(model_cls, "__patched_custom_voice_helper_keys__", False) is False:
            orig_gvc: Callable[..., Any] = model_cls.generate_voice_clone
            VoiceClonePromptItem = q3.VoiceClonePromptItem

            def generate_voice_clone_patched(
                self,
                text,
                language=None,
                ref_audio=None,
                ref_text=None,
                x_vector_only_mode=False,
                voice_clone_prompt=None,
                **kwargs,
            ):
                vc_spk = kwargs.pop("voice_clone_ref_spk_embedding", None)
                if vc_spk is not None:
                    vc_code = kwargs.pop("voice_clone_ref_code", None)
                    vc_xvec = _as_bool(kwargs.pop("voice_clone_x_vector_only_mode", None), default=False)
                    vc_icl = _as_bool(kwargs.pop("voice_clone_icl_mode", None), default=(not vc_xvec))
                    vc_ref_text = _as_str(kwargs.pop("voice_clone_ref_text", None))

                    # If no ref_code is provided, force x-vector only mode.
                    if vc_code is None:
                        vc_xvec = True
                        vc_icl = False

                    # Build prompt items *inside the worker*.
                    voice_clone_prompt = [
                        VoiceClonePromptItem(
                            ref_code=None if vc_xvec else vc_code,
                            ref_spk_embedding=vc_spk,
                            x_vector_only_mode=bool(vc_xvec),
                            icl_mode=bool(vc_icl),
                            ref_text=vc_ref_text,
                        )
                    ]

                    # Avoid slow/ref-audio prompt building.
                    ref_audio = None
                    ref_text = vc_ref_text
                    x_vector_only_mode = bool(vc_xvec)

                return orig_gvc(
                    self,
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    voice_clone_prompt=voice_clone_prompt,
                    **kwargs,
                )

            model_cls.generate_voice_clone = generate_voice_clone_patched  # type: ignore[assignment]
            model_cls.__patched_custom_voice_helper_keys__ = True
    except Exception:
        pass
