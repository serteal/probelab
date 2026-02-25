"""Base class for probes."""

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.optim import AdamW

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
        seed: Random seed for reproducibility. If None, no seeding is done.
        optimizer_fn: Factory ``fn(params) -> optimizer``. If None, uses AdamW.
        scheduler_fn: Factory ``fn(optimizer) -> scheduler``. If None, no scheduler.
    """

    def __init__(
        self,
        device: str | None = None,
        seed: int | None = None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
    ):
        self.device = device
        self.seed = seed
        self._optimizer_fn = optimizer_fn
        self._scheduler_fn = scheduler_fn

    def _seed_everything(self) -> torch.Generator | None:
        """Seed all RNGs for reproducibility.

        Sets global seeds (torch, CUDA, numpy, random) so that weight
        initialisation and dropout are deterministic.  Returns a local
        ``torch.Generator`` that probes can pass to ``randperm`` and
        ``DataLoader`` for deterministic shuffling without further
        mutating global state.  Returns ``None`` when no seed is set.
        """
        if self.seed is None:
            return None
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

    def _make_optimizer(self, params, **defaults):
        """Create optimizer from factory or fall back to AdamW with defaults."""
        if self._optimizer_fn is not None:
            return self._optimizer_fn(params)
        fused = isinstance(self.device, str) and self.device.startswith("cuda")
        return AdamW(params, fused=fused, **defaults)

    def _make_scheduler(self, optimizer):
        """Create scheduler from factory, or return None."""
        if self._scheduler_fn is not None:
            return self._scheduler_fn(optimizer)
        return None

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
