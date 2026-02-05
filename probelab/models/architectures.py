"""Model architecture definitions (functional style).

Each architecture is a frozen dataclass containing callables for accessing
model internals. The get_arch() function auto-detects the architecture.

Note: Tokenizer-related config (prefix patterns, fold_system, token_padding)
is in processing/chat_templates.py - this module only handles model structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PreTrainedModel


@dataclass(frozen=True, slots=True)
class Arch:
    """Architecture specification with callables for model access."""

    get_layers: Callable[[Any], list["nn.Module"]]
    get_layer: Callable[[Any, int], "nn.Module"]
    get_layernorm: Callable[[Any, int], "nn.Module"]
    set_layers: Callable[[Any, list], None]
    num_layers: Callable[[Any], int]

    # Compatibility methods (call the stored callables)
    def get_layer_module(self, model: "PreTrainedModel", layer_idx: int) -> "nn.Module":
        return self.get_layer(model, layer_idx)

    def get_layer_norm(self, model: "PreTrainedModel", layer_idx: int) -> "nn.Module":
        return self.get_layernorm(model, layer_idx)

    def get_num_layers(self, model: "PreTrainedModel") -> int:
        return self.num_layers(model)


# =============================================================================
# Helper functions for num_layers extraction
# =============================================================================


def _num_layers_from_config(model: Any) -> int:
    """Extract num_layers from model config, trying common attribute names."""
    config = model.config
    for attr in ("num_hidden_layers", "n_layers", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot determine number of layers for model: {type(model)}")


def _num_layers_gemma(model: Any) -> int:
    """Extract num_layers for Gemma models (may have language_model wrapper)."""
    base = model.language_model if hasattr(model, "language_model") else model
    return _num_layers_from_config(base)


# =============================================================================
# Architecture definitions
# =============================================================================

ARCHITECTURES: dict[str, Arch] = {
    "llama": Arch(
        get_layers=lambda m: list(m.model.layers),
        get_layer=lambda m, i: m.model.layers[i],
        get_layernorm=lambda m, i: m.model.layers[i].input_layernorm,
        set_layers=lambda m, layers: setattr(m.model, "layers", type(m.model.layers)(layers)),
        num_layers=_num_layers_from_config,
    ),
    "gemma": Arch(
        get_layers=lambda m: list(m.language_model.model.layers),
        get_layer=lambda m, i: m.language_model.model.layers[i],
        get_layernorm=lambda m, i: m.language_model.model.layers[i].input_layernorm,
        set_layers=lambda m, layers: setattr(m.language_model.model, "layers", type(m.language_model.model.layers)(layers)),
        num_layers=_num_layers_gemma,
    ),
    "gemma3": Arch(
        get_layers=lambda m: list(m.language_model.layers),
        get_layer=lambda m, i: m.language_model.layers[i],
        get_layernorm=lambda m, i: m.language_model.layers[i].input_layernorm,
        set_layers=lambda m, layers: setattr(m.language_model, "layers", type(m.language_model.layers)(layers)),
        num_layers=lambda m: m.language_model.config.num_hidden_layers,
    ),
}


# =============================================================================
# Public API
# =============================================================================


def get_arch(model: "PreTrainedModel") -> Arch:
    """Auto-detect and return the architecture for a model.

    Handles PEFT-wrapped models by unwrapping to the base model.
    """
    # Unwrap PEFT models
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    for arch in ARCHITECTURES.values():
        try:
            arch.get_layernorm(base, 0)
            return arch
        except (AttributeError, IndexError, TypeError):
            continue

    supported = ", ".join(ARCHITECTURES.keys())
    raise ValueError(
        f"Unsupported model architecture: {type(base).__name__}\n"
        f"Supported: {supported}"
    )


def get_arch_by_name(name: str) -> Arch:
    """Get architecture by name."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}. Supported: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[name]
