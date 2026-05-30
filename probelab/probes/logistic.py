"""L2-regularized logistic regression probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class Logistic(BaseProbe):
    """L2-regularized logistic regression probe."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 500,
        n_epochs: int = 100,
        batch_size: int = 8192,
        *,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
    ):
        super().__init__(
            device=device,
            seed=seed,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            cast=cast,
        )
        self.C = C
        self.max_iter = max_iter
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.linear: nn.Linear | None = None
        self.register_buffer("scaler_mean", torch.empty(0))
        self.register_buffer("scaler_std", torch.empty(0))

    def initialize(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> "Logistic":
        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or features.device)
        with self._temporary_seed():
            self.linear = nn.Linear(features.shape[-1], 1, bias=True)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
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
        return self.linear(x).squeeze(-1)

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "Logistic":
        features, labels = self._feature_data_from_activations(X, y)
        if self.device is None:
            self.device = str(X.data.device)
        if features.shape[0] == 0:
            raise ValueError(
                f"{self.__class__.__name__}.fit received no training features "
                "(0 samples/tokens after masking). Check the activations and mask."
            )
        if features.shape[0] < 2:
            raise ValueError("Logistic requires at least two training samples or tokens.")
        if torch.unique(labels.detach().cpu()).numel() < 2:
            raise ValueError("Logistic requires both classes to be present in training labels.")
        features = features.to(dtype=self._resolve_dtype(features.dtype))
        labels = labels.to(dtype=features.dtype)
        self.initialize(features, labels)

        n = features.shape[0]
        batch_size = min(kwargs.get("batch_size", self.batch_size), n)
        l2_weight = 1.0 / (2.0 * self.C * n) if self.C > 0 else 0.0
        sc_mean, sc_std = self.scaler_mean, self.scaler_std
        device, dtype = self._module_device_dtype()

        if self._optimizer_fn is not None:
            optimizer = self._optimizer_fn(self.parameters())
            scheduler = self._make_scheduler(optimizer)
            g = self._make_generator()
            with self._temporary_seed():
                self.train()
                for _ in range(self.n_epochs):
                    perm = torch.randperm(n, device="cpu", generator=g).to(features.device)
                    for start in range(0, n, batch_size):
                        idx = perm[start : start + batch_size]
                        x_batch = (features[idx].to(device=device, dtype=dtype) - sc_mean) / sc_std
                        y_batch = labels[idx].to(device=device, dtype=dtype)
                        optimizer.zero_grad()
                        loss = F.binary_cross_entropy_with_logits(self.linear(x_batch).squeeze(-1), y_batch)
                        if l2_weight > 0:
                            loss = loss + l2_weight * self.linear.weight.pow(2).sum()
                        loss.backward()
                        optimizer.step()
                    self._step_scheduler(scheduler)
            self.eval()
            return self

        optimizer = torch.optim.LBFGS(
            self.parameters(), max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            running_loss = 0.0
            for start in range(0, n, batch_size):
                x_chunk = (features[start : start + batch_size].to(device=device, dtype=dtype) - sc_mean) / sc_std
                y_chunk = labels[start : start + batch_size].to(device=device, dtype=dtype)
                chunk_loss = F.binary_cross_entropy_with_logits(
                    self.linear(x_chunk).squeeze(-1),
                    y_chunk,
                    reduction="sum",
                )
                (chunk_loss / n).backward()
                running_loss += chunk_loss.item()
            total = running_loss / n
            if l2_weight > 0:
                l2 = l2_weight * self.linear.weight.pow(2).sum()
                l2.backward()
                total += l2.item()
            return total

        optimizer.step(closure)
        self.eval()
        return self

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
                "C": self.C,
                "max_iter": self.max_iter,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
            },
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Logistic":
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
        return f"Logistic(C={self.C}, fitted={self.fitted})"
