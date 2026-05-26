"""Truncated Polynomial Classifier (TPC).

Progressive polynomial probe with symmetric CP decomposition.
Evaluates degree-by-degree for cascading early-exit.

Reference: Oldfield et al., "Beyond Linear Probes" (2025), arXiv:2509.26238.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class _TPCNetwork(nn.Module):
    """Truncated polynomial with symmetric CP decomposition.

    P(z) = linear(z) + Σ_{k=2}^{N} Σ_{r=1}^{R} λ_r^[k] · (u_r^[k]·z)^k
    """

    def __init__(self, d_model: int, max_degree: int = 3, rank: int = 64):
        super().__init__()
        self.max_degree = max_degree
        self.rank = rank
        # Degree 1: standard linear
        self.linear = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Higher degrees: factor matrices U^[k] and coefficient vectors λ^[k]
        self.factors = nn.ParameterList()
        self.coeffs = nn.ParameterList()
        for k in range(2, max_degree + 1):
            U = nn.Parameter(torch.randn(rank, d_model) * 0.01)
            lam = nn.Parameter(torch.zeros(rank))
            self.factors.append(U)
            self.coeffs.append(lam)

    def forward(self, x: torch.Tensor, max_degree: int | None = None) -> torch.Tensor:
        if max_degree is None:
            max_degree = self.max_degree
        out = self.linear(x).squeeze(-1)  # [batch]
        for k_idx, k in enumerate(range(2, min(max_degree, self.max_degree) + 1)):
            U = self.factors[k_idx]  # [R, d_model]
            lam = self.coeffs[k_idx]  # [R]
            proj = x @ U.T  # [batch, R]
            out = out + (lam * proj.pow(k)).sum(-1)
        return out


class TPC(BaseProbe):
    """Truncated Polynomial Classifier.

    Progressive degree-wise training: freeze lower degrees, train higher.
    Supports cascading early-exit at inference.

    Adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores
    """

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
        self.net = None
        self.scaler_mean = None
        self.scaler_std = None

    @property
    def fitted(self) -> bool:
        return self.net is not None and self.scaler_mean is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "TPC":
        if "l" in X.dims:
            raise ValueError(
                f"TPC expects no LAYER axis. "
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
        sc_mean, sc_std = self.scaler_mean, self.scaler_std

        self._seed_everything()
        self.net = _TPCNetwork(d_model, self.max_degree, self.rank).to(self.device, dtype=working_dtype)

        # Progressive training: degree by degree
        # Degree 1: train linear term
        self._train_degree(features, labels, N, bs, sc_mean, sc_std, degree=1)

        # Higher degrees: freeze lower, train current
        for k in range(2, self.max_degree + 1):
            self._train_degree(features, labels, N, bs, sc_mean, sc_std, degree=k)

        self.net.eval()
        return self

    def _train_degree(self, features, labels, N, bs, sc_mean, sc_std, degree: int):
        """Train parameters for a single degree, freezing all others."""
        # Freeze everything
        for p in self.net.parameters():
            p.requires_grad_(False)

        # Unfreeze only the current degree's parameters
        if degree == 1:
            self.net.linear.weight.requires_grad_(True)
            self.net.linear.bias.requires_grad_(True)
            trainable = [self.net.linear.weight, self.net.linear.bias]
        else:
            k_idx = degree - 2
            self.net.factors[k_idx].requires_grad_(True)
            self.net.coeffs[k_idx].requires_grad_(True)
            trainable = [self.net.factors[k_idx], self.net.coeffs[k_idx]]

        fused = isinstance(self.device, str) and self.device.startswith("cuda")
        optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay,
            fused=fused,
        )

        self.net.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(N, device=features.device)
            for i in range(0, N, bs):
                idx = perm[i : i + bs]
                x_batch = (features[idx].to(self.device) - sc_mean) / sc_std
                optimizer.zero_grad()
                logits = self.net(x_batch, max_degree=degree)
                loss = F.binary_cross_entropy_with_logits(logits, labels[idx].to(self.device))
                loss.backward()
                optimizer.step()

        # Re-freeze what we just trained
        for p in trainable:
            p.requires_grad_(False)

    def __call__(self, x: torch.Tensor, max_degree: int | None = None) -> torch.Tensor:
        self._check_fitted()
        x = x.to(self.device)
        x_scaled = (x - self.scaler_mean) / self.scaler_std
        return self.net(x_scaled, max_degree=max_degree)

    def predict(self, X: Activations) -> torch.Tensor:
        self._check_fitted()
        if "l" in X.dims:
            raise ValueError(f"TPC expects no LAYER axis. Current dims: {X.dims}")

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

    def predict_cascade(self, X: Activations, threshold: float = 0.8) -> torch.Tensor:
        """Cascading inference: evaluate degree-by-degree, early-exit when confident.

        Args:
            X: Activations without LAYER axis (pre-pooled or with SEQ for token-level)
            threshold: Confidence threshold. Exit when sigmoid(logit) > threshold
                       or sigmoid(logit) < (1 - threshold).

        Returns:
            Probabilities tensor.
        """
        self._check_fitted()
        if "l" in X.dims:
            raise ValueError(f"TPC expects no LAYER axis. Current dims: {X.dims}")

        net_dtype = next(self.net.parameters()).dtype

        with torch.no_grad():
            if "s" in X.dims:
                features, _ = X.extract_tokens()
            else:
                features = X.data

            features = features.to(dtype=net_dtype, device=self.device)
            x_scaled = (features - self.scaler_mean) / self.scaler_std
            N = x_scaled.shape[0]
            probs = torch.full((N,), 0.5, device=self.device, dtype=net_dtype)
            remaining = torch.ones(N, dtype=torch.bool, device=self.device)

            for degree in range(1, self.max_degree + 1):
                if not remaining.any():
                    break
                logits = self.net(x_scaled[remaining], max_degree=degree)
                p = torch.sigmoid(logits)
                probs[remaining] = p
                confident = (p > threshold) | (p < (1 - threshold))
                # Mark confident ones as done
                rem_indices = remaining.nonzero(as_tuple=True)[0]
                remaining[rem_indices[confident]] = False

            if "s" in X.dims:
                padded_data, padded_det = X.to_padded()
                out = torch.zeros_like(padded_det, dtype=probs.dtype)
                out[padded_det] = probs
                return out
            return probs

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "max_degree": self.max_degree,
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
    def load(cls, path: Path | str, device: str = "cpu") -> "TPC":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            max_degree=state["max_degree"],
            rank=state["rank"],
            learning_rate=state.get("learning_rate", 1e-3),
            weight_decay=state.get("weight_decay", 1e-2),
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
        probe.net = _TPCNetwork(d_model, probe.max_degree, probe.rank).to(device, dtype=stored_dtype)
        probe.net.load_state_dict(state["network_state"])
        probe.net.eval()
        probe.scaler_mean = state["scaler_mean"].to(device, dtype=stored_dtype)
        probe.scaler_std = state["scaler_std"].to(device, dtype=stored_dtype)
        return probe

    def __repr__(self) -> str:
        return f"TPC(max_degree={self.max_degree}, rank={self.rank}, fitted={self.fitted})"
