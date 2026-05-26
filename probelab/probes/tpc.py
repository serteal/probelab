"""Truncated Polynomial Classifier probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class TPC(BaseProbe):
    """Progressive polynomial probe with symmetric CP decomposition."""

    def __init__(
        self,
        max_degree: int = 3,
        rank: int = 64,
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
        self.max_degree = max_degree
        self.rank = rank
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.linear: nn.Linear | None = None
        self.factors = nn.ParameterList()
        self.coeffs = nn.ParameterList()
        self.register_buffer("scaler_mean", torch.empty(0))
        self.register_buffer("scaler_std", torch.empty(0))

    def initialize(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> "TPC":
        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or features.device)
        with self._temporary_seed():
            d_model = features.shape[-1]
            self.linear = nn.Linear(d_model, 1)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            self.factors = nn.ParameterList()
            self.coeffs = nn.ParameterList()
            for _ in range(2, self.max_degree + 1):
                self.factors.append(nn.Parameter(torch.randn(self.rank, d_model) * 0.01))
                coeff = nn.Parameter(torch.empty(self.rank))
                nn.init.normal_(coeff, mean=0.0, std=0.01)
                self.coeffs.append(coeff)
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

    def forward_degree(self, features: torch.Tensor, degree: int | None = None) -> torch.Tensor:
        self._check_initialized()
        degree = self.max_degree if degree is None else min(degree, self.max_degree)
        x = self._scaled(features)
        out = self.linear(x).squeeze(-1)
        for k_idx, k in enumerate(range(2, degree + 1)):
            proj = x @ self.factors[k_idx].T
            out = out + (self.coeffs[k_idx] * proj.pow(k)).sum(-1)
        return out

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_degree(features, self.max_degree)

    def _train_degree(self, features, labels, degree: int, generator: torch.Generator | None = None):
        for param in self.parameters():
            param.requires_grad_(False)
        if degree == 1:
            trainable = [self.linear.weight, self.linear.bias]
        else:
            k_idx = degree - 2
            trainable = [self.factors[k_idx], self.coeffs[k_idx]]
        for param in trainable:
            param.requires_grad_(True)

        fused = isinstance(self.device, str) and self.device.startswith("cuda")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=fused,
        )
        n = features.shape[0]
        batch_size = min(self.batch_size, n)
        device, dtype = self._module_device_dtype()
        with self._temporary_seed():
            self.train()
            for _ in range(self.n_epochs):
                perm = torch.randperm(n, device="cpu", generator=generator).to(features.device)
                for start in range(0, n, batch_size):
                    idx = perm[start : start + batch_size]
                    optimizer.zero_grad()
                    logits = self.forward_degree(features[idx], degree)
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        labels[idx].to(device=device, dtype=dtype),
                    )
                    loss.backward()
                    optimizer.step()
        for param in trainable:
            param.requires_grad_(False)

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "TPC":
        features, labels = self._feature_data_from_activations(X, y)
        if self.device is None:
            self.device = str(X.data.device)
        if features.shape[0] == 0:
            return self
        features = features.to(dtype=self._resolve_dtype(features.dtype))
        labels = labels.to(dtype=features.dtype)
        self.initialize(features, labels)
        g = self._make_generator()
        for degree in range(1, self.max_degree + 1):
            self._train_degree(features, labels, degree, generator=g)
        for param in self.parameters():
            param.requires_grad_(True)
        self.eval()
        return self

    def predict_logits(self, X: Activations, **kwargs) -> torch.Tensor:
        features, _ = self._feature_data_from_activations(X)
        return self._feature_predict_from_flat(X, self(features))

    def predict(self, X: Activations, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(X, **kwargs))

    def predict_cascade(self, X: Activations, threshold: float = 0.8) -> torch.Tensor:
        self._check_initialized()
        features, _ = self._feature_data_from_activations(X)
        device, dtype = self._module_device_dtype()
        features = features.to(device=device, dtype=dtype)
        probs = torch.full((features.shape[0],), 0.5, device=device, dtype=dtype)
        remaining = torch.ones(features.shape[0], dtype=torch.bool, device=device)
        with torch.no_grad():
            for degree in range(1, self.max_degree + 1):
                if not remaining.any():
                    break
                p = torch.sigmoid(self.forward_degree(features[remaining], degree))
                probs[remaining] = p
                confident = (p > threshold) | (p < (1 - threshold))
                rem_indices = remaining.nonzero(as_tuple=True)[0]
                remaining[rem_indices[confident]] = False
        return self._feature_predict_from_flat(X, probs)

    def save(self, path: Path | str) -> None:
        self._save_probe(
            path,
            {
                "max_degree": self.max_degree,
                "rank": self.rank,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
            },
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "TPC":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["linear.weight"].shape[1]
        probe.initialize(torch.empty(1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"TPC(max_degree={self.max_degree}, rank={self.rank}, fitted={self.fitted})"
