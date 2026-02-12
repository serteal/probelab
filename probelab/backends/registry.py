"""Backend resolution and auto-detection."""

from __future__ import annotations

from typing import Final

from .base import ActivationBackend
from .transformers import TransformersBackend
from .vllm import VLLMBackend

SUPPORTED_BACKENDS: Final[tuple[str, ...]] = ("auto", "transformers", "vllm")


def _is_transformers_model(model_obj: object) -> bool:
    try:
        from transformers import PreTrainedModel
    except Exception:
        return False
    return isinstance(model_obj, PreTrainedModel)


def _is_vllm_activation_engine(model_obj: object) -> bool:
    return hasattr(model_obj, "collect_flat") and hasattr(model_obj, "collect")


def _looks_like_plain_vllm_llm(model_obj: object) -> bool:
    cls = model_obj.__class__
    if cls.__name__ == "LLM" and cls.__module__.startswith("vllm"):
        return True
    return (
        hasattr(model_obj, "generate")
        and hasattr(model_obj, "get_tokenizer")
        and not hasattr(model_obj, "collect_flat")
    )


def resolve_backend_name(model_obj: object, backend: str = "auto") -> str:
    """Resolve backend name from explicit override or object auto-detection."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend={backend!r}. Expected one of: {SUPPORTED_BACKENDS}"
        )

    if backend != "auto":
        if backend == "transformers" and not _is_transformers_model(model_obj):
            raise TypeError(
                "backend='transformers' requires a transformers PreTrainedModel object."
            )
        if backend == "vllm" and not _is_vllm_activation_engine(model_obj):
            raise TypeError(
                "backend='vllm' requires a vllm ActivationEngine-like object "
                "exposing collect_flat(...)."
            )
        return backend

    if _is_transformers_model(model_obj):
        return "transformers"
    if _is_vllm_activation_engine(model_obj):
        return "vllm"
    if _looks_like_plain_vllm_llm(model_obj):
        raise TypeError(
            "Detected plain `vllm.LLM` object. Pass `vllm.activations.ActivationEngine` "
            "for activation collection."
        )
    raise TypeError(
        "Could not auto-detect backend from model object. "
        "Supported: transformers PreTrainedModel or vllm ActivationEngine."
    )


def get_backend(backend_name: str) -> ActivationBackend:
    """Instantiate backend implementation by name."""
    if backend_name == "transformers":
        return TransformersBackend()
    if backend_name == "vllm":
        return VLLMBackend()
    raise ValueError(
        f"Unknown backend={backend_name!r}. Expected one of: {SUPPORTED_BACKENDS[1:]}"
    )


def resolve_backend(model_obj: object, backend: str = "auto") -> ActivationBackend:
    """Resolve and instantiate backend for a model object."""
    return get_backend(resolve_backend_name(model_obj, backend=backend))

