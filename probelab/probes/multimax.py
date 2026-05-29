"""Multi-head hard max pooling probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ..activations import Activations
from .base import BaseProbe


class MultiMax(BaseProbe):
    """Multi-head hard max pooling probe."""

    def __init__(
        self,
        n_heads: int = 10,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
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
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
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
        self.head_projection: nn.Linear | None = None
        self.output: nn.Linear | None = None

    def initialize(self, sequences: torch.Tensor, mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> "MultiMax":
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
            self.head_projection = nn.Linear(self.mlp_hidden_dim, self.n_heads)
            self.output = nn.Linear(self.n_heads, 1)
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
        m = mask.to(device=device).bool()
        mlp_out = self.mlp(self.norm(x))
        scores = self.head_projection(mlp_out).permute(0, 2, 1)
        scores = scores.masked_fill(~m.unsqueeze(1), float("-inf"))
        head_maxes = scores.max(dim=2).values
        head_maxes = torch.nan_to_num(head_maxes, nan=0.0, posinf=0.0, neginf=0.0)
        return self.output(head_maxes).squeeze(-1)

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "MultiMax":
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
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "dropout": self.dropout,
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
    def load(cls, path: Path | str, device: str = "cpu") -> "MultiMax":
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
        return f"MultiMax(n_heads={self.n_heads}, fitted={self.fitted})"
