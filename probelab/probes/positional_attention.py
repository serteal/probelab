"""Positional attention probe.

Heads are intentionally summed with unit weights, matching the original
lightweight positional-attention design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ..activations import Activations
from .base import BaseProbe


class PositionalAttention(BaseProbe):
    """Multi-head cross-attention with learned ALiBi-style bias."""

    def __init__(
        self,
        n_heads: int = 8,
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
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_split = val_split
        self.eval_interval = eval_interval
        self.max_padded_tokens = max_padded_tokens
        self.query_proj: nn.Parameter | None = None
        self.value_proj: nn.Parameter | None = None
        self.position_weights: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None

    def initialize(self, sequences: torch.Tensor, mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> "PositionalAttention":
        working_dtype = self._resolve_dtype(sequences.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or sequences.device)
        with self._temporary_seed():
            d_model = sequences.shape[-1]
            # Zero query/position parameters intentionally start from uniform
            # attention; value projections break symmetry across heads.
            self.query_proj = nn.Parameter(torch.zeros(d_model, self.n_heads))
            self.value_proj = nn.Parameter(torch.empty(d_model, self.n_heads))
            nn.init.xavier_normal_(self.value_proj, gain=0.5)
            self.position_weights = nn.Parameter(torch.zeros(self.n_heads))
            self.bias = nn.Parameter(torch.zeros(1))
        self.to(device=device, dtype=working_dtype)
        self._mark_initialized(working_dtype)
        return self

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = sequences.to(device=device, dtype=dtype)
        m = mask.to(device=device).bool()
        _, seq_len, _ = x.shape
        attn_logits = x @ self.query_proj
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        attn_logits = attn_logits + (positions.unsqueeze(-1) * self.position_weights.unsqueeze(0)).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(~m.unsqueeze(-1), float("-inf"))
        attn_weights = torch.nan_to_num(torch.softmax(attn_logits, dim=1), nan=0.0)
        values = x @ self.value_proj
        # Sum over sequence and heads by design; there is no learned output
        # projection in the original lightweight probe.
        return (attn_weights * values).sum(dim=(1, 2)) + self.bias

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "PositionalAttention":
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
            "n_heads": self.n_heads,
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
    def load(cls, path: Path | str, device: str = "cpu") -> "PositionalAttention":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["query_proj"].shape[0]
        probe.initialize(torch.empty(1, 1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"PositionalAttention(n_heads={self.n_heads}, fitted={self.fitted})"
