"""Normalization utility for activations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..processing.activations import Activations


@dataclass
class Normalize:
    """Standardize features to zero mean, unit variance.

    Example:
        >>> norm = Normalize().fit(train_acts)
        >>> train_acts = norm(train_acts)
        >>> test_acts = norm(test_acts)
    """

    eps: float = 1e-8
    mean_: torch.Tensor | None = field(default=None, repr=False)
    std_: torch.Tensor | None = field(default=None, repr=False)

    def fit(self, X: Activations, y=None) -> "Normalize":
        """Compute mean and std from activations."""
        axes = tuple(range(X.activations.ndim - 1))
        self.mean_ = X.activations.mean(dim=axes, keepdim=True)
        self.std_ = X.activations.std(dim=axes, keepdim=True).clamp(min=self.eps)
        return self

    def __call__(self, X: "Activations") -> "Activations":
        """Apply normalization."""
        from ..processing.activations import Activations

        if self.mean_ is None:
            raise ValueError("Normalize must be fit before calling")
        normalized = (X.activations - self.mean_) / self.std_
        return Activations(
            activations=normalized,
            axes=X.axes,
            layer_meta=X.layer_meta,
            sequence_meta=X.sequence_meta,
            batch_indices=X.batch_indices,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Normalize) and self.eps == other.eps

    def __hash__(self) -> int:
        return hash(("Normalize", self.eps))
