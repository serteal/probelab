"""Multi-head self-attention probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ..activations import Activations
from .base import BaseProbe


class MHA(BaseProbe):
    """Small transformer encoder probe with a learned CLS token."""

    def __init__(
        self,
        proj_dim: int = 128,
        n_heads: int = 4,
        n_enc_layers: int = 1,
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
        self.proj_dim = proj_dim
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_split = val_split
        self.eval_interval = eval_interval
        self.max_padded_tokens = max_padded_tokens
        self.down_proj: nn.Linear | None = None
        self.cls_token: nn.Parameter | None = None
        self.encoder: nn.TransformerEncoder | None = None
        self.classifier: nn.Linear | None = None

    def initialize(self, sequences: torch.Tensor, mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> "MHA":
        working_dtype = self._resolve_dtype(sequences.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or sequences.device)
        with self._temporary_seed():
            d_model = sequences.shape[-1]
            self.down_proj = nn.Linear(d_model, self.proj_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.proj_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=self.n_heads,
                dim_feedforward=4 * self.proj_dim,
                dropout=self.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.n_enc_layers,
                enable_nested_tensor=False,
            )
            self.classifier = nn.Linear(self.proj_dim, 1)
            nn.init.xavier_uniform_(self.down_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        self.to(device=device, dtype=working_dtype)
        self._mark_initialized(working_dtype)
        return self

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = self.down_proj(sequences.to(device=device, dtype=dtype))
        m = mask.to(device=device)
        batch = x.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(batch, 1, dtype=m.dtype, device=device)
        ext_mask = torch.cat([cls_mask, m], dim=1)
        x = self.encoder(x, src_key_padding_mask=~ext_mask.bool())
        return self.classifier(x[:, 0]).squeeze(-1)

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "MHA":
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
            "proj_dim": self.proj_dim,
            "n_heads": self.n_heads,
            "n_enc_layers": self.n_enc_layers,
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
    def load(cls, path: Path | str, device: str = "cpu") -> "MHA":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["down_proj.weight"].shape[1]
        probe.initialize(torch.empty(1, 1, d_model, dtype=dtype, device=device))
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"MHA(proj_dim={self.proj_dim}, n_heads={self.n_heads}, fitted={self.fitted})"
