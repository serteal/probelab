"""Shared types for activation collection adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ActivationChunk:
    """A backend-neutral chunk of flat activations."""

    data: torch.Tensor
    detection_mask: torch.Tensor
    offsets: torch.Tensor
    indices: list[int]
    layers: tuple[int, ...] | None = None
