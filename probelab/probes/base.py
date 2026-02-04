"""Base class for probes."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from ..processing.activations import Activations
from ..processing.scores import Scores


class BaseProbe(ABC):
    """Base class for probes.

    Probes are classifiers that operate on Activations and return Scores.
    They adapt based on input dimensionality:
    - If activations have SEQ axis: Train/predict on tokens
    - If no SEQ axis: Train/predict on sequences
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self._fitted = False

    @abstractmethod
    def fit(self, X: Activations, y: list | torch.Tensor) -> "BaseProbe":
        """Fit probe on activations and labels."""
        pass

    @abstractmethod
    def predict(self, X: Activations) -> Scores:
        """Predict, returning Scores object."""
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save probe to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str, device: str = "cuda") -> "BaseProbe":
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self._fitted})"
