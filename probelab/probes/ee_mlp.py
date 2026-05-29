"""Early-exit MLP probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class EEMLP(BaseProbe):
    """Multi-layer MLP with output heads at intermediate layers."""

    def __init__(
        self,
        n_layers: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
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
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.register_buffer("scaler_mean", torch.empty(0))
        self.register_buffer("scaler_std", torch.empty(0))

    def initialize(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> "EEMLP":
        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or features.device)
        with self._temporary_seed():
            d_model = features.shape[-1]
            dims = [d_model] + [self.hidden_dim] * self.n_layers
            self.layers = nn.ModuleList()
            self.heads = nn.ModuleList([nn.Linear(d_model, 1)])
            for i in range(self.n_layers):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(dims[i], dims[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                    )
                )
                self.heads.append(nn.Linear(dims[i + 1], 1))
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.to(device=device, dtype=working_dtype)
        stats = features.to(dtype=working_dtype)
        self.scaler_mean = stats.mean(0).to(device)
        self.scaler_std = stats.std(0, unbiased=False).clamp(min=1e-8).to(device)
        self._mark_initialized(working_dtype)
        return self

    def _scaled(self, features: torch.Tensor) -> torch.Tensor:
        device, dtype = self._module_device_dtype()
        x = features.to(device=device, dtype=dtype)
        return (x - self.scaler_mean) / self.scaler_std

    def forward_all_exits(self, features: torch.Tensor) -> list[torch.Tensor]:
        self._check_initialized()
        logits_list: list[torch.Tensor] = []
        h = self._scaled(features)
        logits_list.append(self.heads[0](h).squeeze(-1))
        for i, layer in enumerate(self.layers):
            h = layer(h)
            logits_list.append(self.heads[i + 1](h).squeeze(-1))
        return logits_list

    def forward_exit(self, features: torch.Tensor, exit_at: int) -> torch.Tensor:
        exits = self.forward_all_exits(features)
        return exits[min(exit_at, len(exits) - 1)]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_all_exits(features)[-1]

    def loss_on_batch(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        all_logits = self.forward_all_exits(features)
        return sum(F.binary_cross_entropy_with_logits(logits, labels) for logits in all_logits) / len(all_logits)

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "EEMLP":
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
        return self._fit_feature_default(features, labels, **kwargs)

    def predict_logits(self, X: Activations, **kwargs) -> torch.Tensor:
        self._check_initialized()
        features, _ = self._feature_data_from_activations(X)
        logits = self._feature_logits_batched(features, kwargs.get("batch_size"))
        return self._feature_predict_from_flat(X, logits)

    def predict(self, X: Activations, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(X, **kwargs))

    def save(self, path: Path | str) -> None:
        self._save_probe(
            path,
            {
                "n_layers": self.n_layers,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
            },
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "EEMLP":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["heads.0.weight"].shape[1]
        probe.initialize(torch.empty(1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"EEMLP(n_layers={self.n_layers}, hidden_dim={self.hidden_dim}, fitted={self.fitted})"
