"""Attention-based probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ..activations import Activations
from .base import BaseProbe


class Attention(BaseProbe):
    """Learned attention pooling over sequence activations."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        temperature: float = 2.0,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 1000,
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
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_split = val_split
        self.eval_interval = eval_interval
        self.max_padded_tokens = max_padded_tokens
        self.attention_norm: nn.LayerNorm | None = None
        self.attention_scorer: nn.Sequential | None = None
        self.classifier_norm: nn.LayerNorm | None = None
        self.classifier: nn.Sequential | None = None

    def initialize(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> "Attention":
        working_dtype = self._resolve_dtype(sequences.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or sequences.device)
        with self._temporary_seed():
            d_model = sequences.shape[-1]
            self.attention_norm = nn.LayerNorm(d_model)
            self.attention_scorer = nn.Sequential(
                nn.Linear(d_model, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            self.classifier_norm = nn.LayerNorm(d_model)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.to(device=device, dtype=working_dtype)
        self._mark_initialized(working_dtype)
        return self

    def attention_weights(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = sequences.to(device=device, dtype=dtype)
        m = mask.to(device=device).bool()
        scores = self.attention_scorer(self.attention_norm(x)).squeeze(-1) / self.temperature
        scores = scores.masked_fill(~m, float("-inf"))
        return torch.nan_to_num(torch.softmax(scores, dim=1), nan=0.0)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = sequences.to(device=device, dtype=dtype)
        weights = self.attention_weights(x, mask)
        aggregated = (weights.unsqueeze(-1) * x).sum(dim=1)
        return self.classifier(self.classifier_norm(aggregated)).squeeze(-1)

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "Attention":
        self._require_sequence_activations(X)
        if self.device is None:
            self.device = str(X.data.device)
        labels = self._to_labels(y)
        working_dtype = self._resolve_dtype(X.data.dtype)
        dummy = X.data[:1].reshape(1, 1, X.hidden_size).to(dtype=working_dtype)
        self.initialize(dummy)
        return self._fit_sequence_default(X, labels, **kwargs)

    def predict_logits(self, X: Activations, **kwargs) -> torch.Tensor:
        self._require_sequence_activations(X)
        opts = self._fit_kwargs(**kwargs)
        return self._sequence_logits(
            X,
            list(range(X.batch_size)),
            batch_size=opts["batch_size"],
            max_padded_tokens=opts["max_padded_tokens"],
        )

    def predict(self, X: Activations, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(X, **kwargs))

    def save(self, path: Path | str) -> None:
        self._save_probe(
            path,
            {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "temperature": self.temperature,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs": self.n_epochs,
                "patience": self.patience,
                "batch_size": self.batch_size,
                "val_split": self.val_split,
                "eval_interval": self.eval_interval,
                "max_padded_tokens": self.max_padded_tokens,
            },
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Attention":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["attention_norm.weight"].shape[0]
        probe.initialize(torch.empty(1, 1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"Attention(hidden_dim={self.hidden_dim}, fitted={self.fitted})"
