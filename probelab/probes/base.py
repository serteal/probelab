"""Base class for probes."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

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
        self.device = device

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Whether the probe has been fitted. Each probe defines its own check."""
        ...

    @abstractmethod
    def fit(self, X: Activations, y: list | torch.Tensor) -> "BaseProbe":
        """Fit probe on activations and labels."""
        ...

    @abstractmethod
    def predict(self, X: Activations) -> torch.Tensor:
        """Predict probabilities from Activations."""
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save probe to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "BaseProbe":
        """Load probe from disk."""
        ...

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before predict")

    def _to_labels(self, y) -> torch.Tensor:
        """Convert labels list to tensor."""
        if isinstance(y, torch.Tensor):
            return y
        return torch.tensor([l.value if hasattr(l, "value") else l for l in y])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.fitted})"
