"""Early-exit MLP probe.

Multi-layer MLP with output heads at each intermediate layer.
Joint training of all heads enables cascading early-exit at inference.

Reference: Oldfield et al., "Beyond Linear Probes" (2025), arXiv:2509.26238.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class _EEMLPNetwork(nn.Module):
    """Multi-layer MLP with early-exit output heads."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        dims = [d_model] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        # Exit head 0: linear probe on raw input (before any hidden layers)
        self.heads.append(nn.Linear(d_model, 1))
        # Hidden layers + exit heads 1..n_layers
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            self.heads.append(nn.Linear(dims[i + 1], 1))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, exit_at: int | None = None) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features [batch, d_model]
            exit_at: If set, return logits from this exit point (0 = raw input,
                     1..n_layers = after each hidden layer).
                     If None, return list of logits from all exit points.

        Returns:
            Single logit tensor [batch] if exit_at set, else list of [batch] tensors.
        """
        logits_list = []
        h = x
        # Exit head 0: linear probe on raw input
        logits = self.heads[0](h).squeeze(-1)
        if exit_at is not None and exit_at == 0:
            return logits
        logits_list.append(logits)
        # Exit heads 1..n_layers: after each hidden layer
        for i, layer in enumerate(self.layers):
            h = layer(h)
            logits = self.heads[i + 1](h).squeeze(-1)
            if exit_at is not None and exit_at == i + 1:
                return logits
            logits_list.append(logits)
        if exit_at is not None:
            return logits_list[-1]
        return logits_list


class EEMLP(BaseProbe):
    """Early-exit MLP probe.

    Trains all exit heads jointly. At inference, uses the deepest head by default.

    Adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores
    """

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
        self.net = None
        self.scaler_mean = None
        self.scaler_std = None

    @property
    def fitted(self) -> bool:
        return self.net is not None and self.scaler_mean is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "EEMLP":
        if "l" in X.dims:
            raise ValueError(
                f"EEMLP expects no LAYER axis. "
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

        g = self._seed_everything()
        self.net = _EEMLPNetwork(d_model, self.n_layers, self.hidden_dim, self.dropout).to(
            self.device, dtype=working_dtype
        )

        optimizer = self._make_optimizer(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        scheduler = self._make_scheduler(optimizer)

        self.net.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(N, device=features.device)
            for i in range(0, N, bs):
                idx = perm[i : i + bs]
                x_batch = (features[idx].to(self.device) - sc_mean) / sc_std
                y_batch = labels[idx].to(self.device)
                optimizer.zero_grad()
                # Joint loss: mean of BCE across all exit heads
                all_logits = self.net(x_batch)
                loss = sum(
                    F.binary_cross_entropy_with_logits(logits, y_batch)
                    for logits in all_logits
                ) / len(all_logits)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        self.net.eval()
        return self

    def __call__(self, x: torch.Tensor, exit_at: int | None = None) -> torch.Tensor:
        self._check_fitted()
        x = x.to(self.device)
        x_scaled = (x - self.scaler_mean) / self.scaler_std
        result = self.net(x_scaled, exit_at=exit_at)
        if isinstance(result, list):
            return result[-1]  # deepest head
        return result

    def predict(self, X: Activations) -> torch.Tensor:
        """Predict using the deepest exit head."""
        self._check_fitted()
        if "l" in X.dims:
            raise ValueError(f"EEMLP expects no LAYER axis. Current dims: {X.dims}")

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
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
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
    def load(cls, path: Path | str, device: str = "cpu") -> "EEMLP":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            n_layers=state["n_layers"],
            hidden_dim=state["hidden_dim"],
            dropout=state.get("dropout", 0.1),
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
        probe.net = _EEMLPNetwork(d_model, probe.n_layers, probe.hidden_dim, probe.dropout).to(
            device, dtype=stored_dtype
        )
        probe.net.load_state_dict(state["network_state"])
        probe.net.eval()
        probe.scaler_mean = state["scaler_mean"].to(device, dtype=stored_dtype)
        probe.scaler_std = state["scaler_std"].to(device, dtype=stored_dtype)
        return probe

    def __repr__(self) -> str:
        return f"EEMLP(n_layers={self.n_layers}, hidden_dim={self.hidden_dim}, fitted={self.fitted})"
