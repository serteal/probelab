"""Low-rank bilinear probe.

Quadratic classifier: y = Σ_r λ_r·(u_r·z)² + b  (symmetric CP rank-1 form).
Learned per-rank coefficients allow the effective weight matrix to have
negative eigenvalues, unlike a plain ||Az||² form.

Reference: Oldfield et al., "Beyond Linear Probes" (2025), arXiv:2509.26238.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..processing.activations import Activations
from .base import BaseProbe


class _BilinearNetwork(nn.Module):
    """Symmetric CP quadratic classifier: y = Σ_r λ_r·(u_r·z)² + b."""

    def __init__(self, d_model: int, rank: int = 32):
        super().__init__()
        self.U = nn.Linear(d_model, rank, bias=False)
        self.lam = nn.Parameter(torch.zeros(rank))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_normal_(self.U.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.U(x)  # [batch, rank]
        return (self.lam * proj.pow(2)).sum(-1) + self.bias


class Bilinear(BaseProbe):
    """Low-rank bilinear probe.

    Adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores
    """

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
        self.net = None
        self.scaler_mean = None
        self.scaler_std = None

    @property
    def fitted(self) -> bool:
        return self.net is not None and self.scaler_mean is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "Bilinear":
        if "l" in X.dims:
            raise ValueError(
                f"Bilinear expects no LAYER axis. "
                f'Call select("l", layer) first. Current dims: {X.dims}'
            )

        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y).to(self.device)

        if "s" in X.dims:
            features, tokens_per_sample = X.extract_tokens()
            labels = torch.repeat_interleave(y_tensor, tokens_per_sample.to(y_tensor.device))
        else:
            features = X.data
            labels = y_tensor

        if features.shape[0] == 0:
            return self

        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype

        features = features.to(dtype=working_dtype)
        labels = labels.to(dtype=working_dtype)
        d_model = features.shape[1]
        N = features.shape[0]
        bs = min(self.batch_size, N)

        self.scaler_mean = features.mean(0).to(self.device)
        self.scaler_std = features.std(0).clamp(min=1e-8).to(self.device)

        g = self._seed_everything()
        self.net = _BilinearNetwork(d_model, self.rank).to(self.device, dtype=working_dtype)

        optimizer = self._make_optimizer(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        scheduler = self._make_scheduler(optimizer)

        sc_mean, sc_std = self.scaler_mean, self.scaler_std
        self.net.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(N, device=features.device)
            for i in range(0, N, bs):
                idx = perm[i : i + bs]
                x_batch = (features[idx].to(self.device) - sc_mean) / sc_std
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    self.net(x_batch), labels[idx].to(self.device)
                )
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        self.net.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        x = x.to(self.device)
        x_scaled = (x - self.scaler_mean) / self.scaler_std
        return self.net(x_scaled)

    def predict(self, X: Activations) -> torch.Tensor:
        self._check_fitted()
        if "l" in X.dims:
            raise ValueError(f"Bilinear expects no LAYER axis. Current dims: {X.dims}")

        net_dtype = next(self.net.parameters()).dtype

        with torch.no_grad():
            if "s" in X.dims:
                features, _ = X.extract_tokens()
                flat_probs = torch.sigmoid(self(features.to(dtype=net_dtype)))
                padded_data, padded_det = X.to_padded()
                probs = torch.zeros_like(padded_det, dtype=flat_probs.dtype)
                probs[padded_det] = flat_probs
                return probs
            else:
                return torch.sigmoid(self(X.data.to(dtype=net_dtype)))

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": self.device,
            "cast": self.cast,
            "training_dtype": str(self._training_dtype),
            "network_state": self.net.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Bilinear":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            rank=state["rank"],
            learning_rate=state.get("learning_rate", 1e-3),
            weight_decay=state.get("weight_decay", 1e-3),
            n_epochs=state["n_epochs"],
            batch_size=state.get("batch_size", 1024),
            seed=state.get("seed"),
            device=device,
            cast=state.get("cast"),
        )
        dtype_str = state.get("training_dtype")
        stored_dtype = getattr(torch, dtype_str.split(".")[-1]) if dtype_str else torch.float32
        probe._training_dtype = stored_dtype

        d_model = state["scaler_mean"].shape[0]
        probe.net = _BilinearNetwork(d_model, probe.rank).to(device, dtype=stored_dtype)
        probe.net.load_state_dict(state["network_state"])
        probe.net.eval()
        probe.scaler_mean = state["scaler_mean"].to(device, dtype=stored_dtype)
        probe.scaler_std = state["scaler_std"].to(device, dtype=stored_dtype)
        return probe

    def __repr__(self) -> str:
        return f"Bilinear(rank={self.rank}, fitted={self.fitted})"
