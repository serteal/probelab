"""Gated bipolar sequence probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class GatedBipolar(BaseProbe):
    """Gated projections with max and negative-min pooling."""

    def __init__(
        self,
        mlp_hidden_dim: int = 128,
        gate_dim: int = 64,
        dropout: float = 0.1,
        lambda_l1: float = 1e-5,
        lambda_orth: float = 1e-4,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 20,
        patience: int = 5,
        batch_size: int = 32,
        val_split: float = 0.2,
        eval_interval: int = 1,
        max_padded_tokens: int | None = None,
        *,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
    ):
        super().__init__(device=device, seed=seed, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn, cast=cast)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.gate_dim = gate_dim
        self.dropout = dropout
        self.lambda_l1 = lambda_l1
        self.lambda_orth = lambda_orth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_split = val_split
        self.eval_interval = eval_interval
        self.max_padded_tokens = max_padded_tokens
        self.norm: nn.LayerNorm | None = None
        self.mlp: nn.Sequential | None = None
        self.W_proj: nn.Linear | None = None
        self.W_gate: nn.Linear | None = None
        self.output: nn.Linear | None = None

    def initialize(self, sequences: torch.Tensor, mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> "GatedBipolar":
        working_dtype = self._resolve_dtype(sequences.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or sequences.device)
        with self._temporary_seed():
            d_model = sequences.shape[-1]
            self.norm = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, self.mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            )
            self.W_proj = nn.Linear(self.mlp_hidden_dim, self.gate_dim)
            self.W_gate = nn.Linear(self.mlp_hidden_dim, self.gate_dim)
            self.output = nn.Linear(2 * self.gate_dim, 1)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.to(device=device, dtype=working_dtype)
        self._mark_initialized(working_dtype)
        return self

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = sequences.to(device=device, dtype=dtype)
        m = mask.to(device=device).bool().unsqueeze(-1)
        h = self.mlp(self.norm(x))
        v = self.W_proj(h) * F.softplus(self.W_gate(h))
        v_max = v.masked_fill(~m, float("-inf")).max(dim=1).values
        v_min = v.masked_fill(~m, float("inf")).min(dim=1).values
        v_max = torch.nan_to_num(v_max, nan=0.0, posinf=0.0, neginf=0.0)
        v_min = torch.nan_to_num(v_min, nan=0.0, posinf=0.0, neginf=0.0)
        return self.output(torch.cat([v_max, -v_min], dim=-1)).squeeze(-1)

    def regularization_loss(self) -> torch.Tensor:
        self._check_initialized()
        w = self.W_proj.weight
        l1 = self.lambda_l1 * w.abs().sum()
        wtw = w @ w.T
        eye = torch.eye(self.gate_dim, device=w.device, dtype=w.dtype)
        return l1 + self.lambda_orth * (wtw - eye).pow(2).sum()

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "GatedBipolar":
        self._require_sequence_activations(X)
        if self.device is None:
            self.device = str(X.data.device)
        self.initialize(X.data[:1].reshape(1, 1, X.hidden_size).to(dtype=self._resolve_dtype(X.data.dtype)))
        return self._fit_sequence_default(X, self._to_labels(y), **kwargs)

    def predict_logits(self, X: Activations, **kwargs) -> torch.Tensor:
        self._require_sequence_activations(X)
        opts = self._fit_kwargs(**kwargs)
        return self._sequence_logits(X, list(range(X.batch_size)), batch_size=opts["batch_size"], max_padded_tokens=opts["max_padded_tokens"])

    def predict(self, X: Activations, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(X, **kwargs))

    def save(self, path: Path | str) -> None:
        self._save_probe(path, {
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "gate_dim": self.gate_dim,
            "dropout": self.dropout,
            "lambda_l1": self.lambda_l1,
            "lambda_orth": self.lambda_orth,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "val_split": self.val_split,
            "eval_interval": self.eval_interval,
            "max_padded_tokens": self.max_padded_tokens,
        })

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "GatedBipolar":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["norm.weight"].shape[0]
        probe.initialize(torch.empty(1, 1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"GatedBipolar(gate_dim={self.gate_dim}, fitted={self.fitted})"
