"""Multi-layer perceptron probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn

from ..activations import Activations
from .base import BaseProbe


class MLP(BaseProbe):
    """Single-hidden-layer MLP probe."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float | None = None,
        activation: Literal["relu", "gelu"] = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        *,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
    ):
        super().__init__(device=device, seed=seed, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn, cast=cast)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fc1: nn.Linear | None = None
        self.activation_layer: nn.Module | None = None
        self.dropout_layer: nn.Module | None = None
        self.fc2: nn.Linear | None = None

    def initialize(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> "MLP":
        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or features.device)
        with self._temporary_seed():
            self.fc1 = nn.Linear(features.shape[-1], self.hidden_dim)
            self.activation_layer = nn.ReLU() if self.activation == "relu" else nn.GELU()
            self.dropout_layer = nn.Dropout(self.dropout) if self.dropout is not None else None
            self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.to(device=device, dtype=working_dtype)
        self._mark_initialized(working_dtype)
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = features.to(device=device, dtype=dtype)
        x = self.fc1(x)
        x = self.activation_layer(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return self.fc2(x).squeeze(-1)

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "MLP":
        features, labels = self._feature_data_from_activations(X, y)
        if self.device is None:
            self.device = str(X.data.device)
        if features.shape[0] == 0:
            raise ValueError(
                f"{self.__class__.__name__}.fit received no training features "
                "(0 samples/tokens after masking). Check the activations and mask."
            )
        features = features.to(dtype=self._resolve_dtype(features.dtype))
        labels = labels.to(dtype=features.dtype)
        self.initialize(features, labels)
        return self._fit_feature_default(features, labels, dataloader=True, **kwargs)

    def predict_logits(self, X: Activations, **kwargs) -> torch.Tensor:
        self._check_initialized()
        features, _ = self._feature_data_from_activations(X)
        logits = self._feature_logits_batched(features, kwargs.get("batch_size"))
        return self._feature_predict_from_flat(X, logits)

    def predict(self, X: Activations, **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(X, **kwargs))

    def save(self, path: Path | str) -> None:
        self._save_probe(
            path,
            {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "activation": self.activation,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
            },
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MLP":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["fc1.weight"].shape[1]
        probe.initialize(torch.empty(1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"MLP(hidden_dim={self.hidden_dim}, fitted={self.fitted})"
