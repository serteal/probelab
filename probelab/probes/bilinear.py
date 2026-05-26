"""Low-rank bilinear probe.

This is intentionally a pure symmetric CP quadratic classifier with no linear
term. Use TPC when a linear term plus higher-order polynomial terms is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ..activations import Activations
from .base import BaseProbe


class Bilinear(BaseProbe):
    """Symmetric CP quadratic classifier."""

    def __init__(
        self,
        rank: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 1024,
        *,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
    ):
        super().__init__(device=device, seed=seed, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn, cast=cast)
        self.rank = rank
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.U: nn.Linear | None = None
        self.lam: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None
        self.register_buffer("scaler_mean", torch.empty(0))
        self.register_buffer("scaler_std", torch.empty(0))

    def initialize(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> "Bilinear":
        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or features.device)
        with self._temporary_seed():
            self.U = nn.Linear(features.shape[-1], self.rank, bias=False)
            self.lam = nn.Parameter(torch.empty(self.rank))
            self.bias = nn.Parameter(torch.zeros(1))
            nn.init.xavier_normal_(self.U.weight, gain=0.5)
            nn.init.normal_(self.lam, mean=0.0, std=0.01)
        self.to(device=device, dtype=working_dtype)
        stats = features.to(dtype=working_dtype)
        self.scaler_mean = stats.mean(0).to(device)
        self.scaler_std = stats.std(0, unbiased=False).clamp(min=1e-8).to(device)
        self._mark_initialized(working_dtype)
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = features.to(device=device, dtype=dtype)
        x = (x - self.scaler_mean) / self.scaler_std
        proj = self.U(x)
        return (self.lam * proj.pow(2)).sum(-1) + self.bias

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "Bilinear":
        features, labels = self._feature_data_from_activations(X, y)
        if self.device is None:
            self.device = str(X.data.device)
        if features.shape[0] == 0:
            return self
        features = features.to(dtype=self._resolve_dtype(features.dtype))
        labels = labels.to(dtype=features.dtype)
        self.initialize(features, labels)
        return self._fit_feature_default(features, labels, shuffle_with_generator=False, **kwargs)

    def predict_logits(self, X: Activations, **kwargs) -> torch.Tensor:
        self._check_initialized()
        features, _ = self._feature_data_from_activations(X)
        return self._feature_predict_from_flat(X, self(features))

    def predict(self, X: Activations, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(X, **kwargs))

    def save(self, path: Path | str) -> None:
        self._save_probe(
            path,
            {
                "rank": self.rank,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
            },
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Bilinear":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["U.weight"].shape[1]
        probe.initialize(torch.empty(1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"Bilinear(rank={self.rank}, fitted={self.fitted})"
