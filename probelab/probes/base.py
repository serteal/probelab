"""Base class for probes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from ..processing.acts import Acts
from ..processing.activations import Activations


class BaseProbe(ABC):
    """Base class for probes.

    Probes are classifiers that operate on Activations and return probability tensors.
    They adapt based on input dimensionality:
    - If activations have SEQ axis: Train/predict on tokens
    - If no SEQ axis: Train/predict on sequences

    Two interfaces:
    - probe(x): Differentiable forward pass on raw tensors, returns logits
    - probe.predict(X): Convenience method for Activations, returns probabilities

    Args:
        device: Device to use. If None, auto-detects from input in fit().
    """

    def __init__(self, device: str | None = None):
        self.device = device  # None = auto-detect from input
        self._fitted = False

    @abstractmethod
    def fit(self, X: Activations | Acts | torch.Tensor, y: list | torch.Tensor) -> "BaseProbe":
        """Fit probe on activations and labels."""
        pass

    @abstractmethod
    def predict(self, X: Activations | Acts | torch.Tensor) -> torch.Tensor:
        """Predict probabilities from Activations.

        Returns:
            Tensor of shape [batch, 2] or [n_tokens, 2] with probabilities.
        """
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save probe to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "BaseProbe":
        """Load probe from disk."""
        pass

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before predict")

    def _to_labels(self, y) -> torch.Tensor:
        """Convert labels list to tensor."""
        if isinstance(y, torch.Tensor):
            return y
        return torch.tensor([l.value if hasattr(l, "value") else l for l in y])

    def _to_activations(self, X: Activations | Acts | torch.Tensor) -> Activations:
        """Normalize heterogeneous probe inputs to Activations for compatibility paths."""
        from ..processing.activations import Axis, LayerMeta, SequenceMeta

        if isinstance(X, Activations):
            return X
        if isinstance(X, Acts):
            return X.to_activations()
        if isinstance(X, torch.Tensor):
            if X.ndim == 2:
                return Activations.from_tensor(X)
            if X.ndim == 3:
                b, s, _ = X.shape
                return Activations(
                    activations=X,
                    axes=(Axis.BATCH, Axis.SEQ, Axis.HIDDEN),
                    layer_meta=None,
                    sequence_meta=SequenceMeta(
                        attention_mask=torch.ones(b, s, device=X.device),
                        detection_mask=torch.ones(b, s, device=X.device),
                        input_ids=torch.ones(b, s, device=X.device, dtype=torch.long),
                    ),
                    batch_indices=torch.arange(b, device=X.device),
                )
            if X.ndim == 4:
                b, l, s, _ = X.shape
                return Activations(
                    activations=X,
                    axes=(Axis.BATCH, Axis.LAYER, Axis.SEQ, Axis.HIDDEN),
                    layer_meta=LayerMeta(tuple(range(l))),
                    sequence_meta=SequenceMeta(
                        attention_mask=torch.ones(b, s, device=X.device),
                        detection_mask=torch.ones(b, s, device=X.device),
                        input_ids=torch.ones(b, s, device=X.device, dtype=torch.long),
                    ),
                    batch_indices=torch.arange(b, device=X.device),
                )
        raise TypeError(f"Unsupported input type for probes: {type(X).__name__}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self._fitted})"
