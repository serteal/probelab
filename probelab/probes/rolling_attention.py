"""Rolling attention probe (Max-of-Rolling-Means).

Per-head soft attention within sliding windows, max across windows.
A "soft MultiMax" variant.

Reference: Kramár et al., "Building Production-Ready Probes for Gemini" (2026).
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..processing.activations import Activations
from .base import BaseProbe


class _RollingAttentionNetwork(nn.Module):
    """Per-head windowed attention with max-across-windows pooling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 10,
        hidden_dim: int = 128,
        window_size: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Per-head query and value vectors
        self.queries = nn.Parameter(torch.empty(n_heads, hidden_dim))
        self.values = nn.Parameter(torch.empty(n_heads, hidden_dim))
        nn.init.xavier_normal_(self.queries, gain=0.5)
        nn.init.xavier_normal_(self.values, gain=0.5)

        self.output = nn.Linear(n_heads, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, _ = sequences.shape
        w = self.window_size

        y = self.mlp(self.norm(sequences))  # [B, S, hidden_dim]

        # Per-head attention logits and values
        # logits: [B, S, H], vals: [B, S, H]
        logits = torch.einsum("bsd,hd->bsh", y, self.queries)
        vals = torch.einsum("bsd,hd->bsh", y, self.values)

        # Mask invalid positions
        mask_bool = mask.bool()  # [B, S]
        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))

        # If sequence shorter than window, use full sequence as single window
        if S <= w:
            attn = torch.softmax(logits, dim=1)
            attn = torch.nan_to_num(attn, nan=0.0)
            mask_f = mask_bool.unsqueeze(-1).to(attn.dtype)
            weighted = (attn * vals * mask_f).sum(dim=1)  # [B, H]
            return self.output(weighted).squeeze(-1)

        # Sliding window via unfold: [B, S, H] → [B, n_windows, w, H]
        n_windows = S - w + 1
        logits_win = logits.unfold(1, w, 1)   # [B, n_windows, H, w]
        vals_win = vals.unfold(1, w, 1)        # [B, n_windows, H, w]
        mask_win = mask_bool.unfold(1, w, 1)   # [B, n_windows, w]

        # Softmax within each window (over the w dimension, last dim)
        logits_win = logits_win.masked_fill(~mask_win.unsqueeze(2), float("-inf"))
        attn_win = torch.softmax(logits_win, dim=-1)  # [B, n_windows, H, w]
        attn_win = torch.nan_to_num(attn_win, nan=0.0)

        # Weighted mean of values within each window
        # [B, n_windows, H, w] * [B, n_windows, H, w] → sum → [B, n_windows, H]
        window_means = (attn_win * vals_win).sum(dim=-1)

        # Check if window has any valid tokens
        valid_windows = mask_win.any(dim=-1)  # [B, n_windows]
        window_means = window_means.masked_fill(~valid_windows.unsqueeze(-1), float("-inf"))

        # Max across windows per head
        head_maxes = window_means.max(dim=1).values  # [B, H]
        head_maxes = torch.nan_to_num(head_maxes, nan=0.0, posinf=0.0, neginf=0.0)

        return self.output(head_maxes).squeeze(-1)  # [B]


class RollingAttention(BaseProbe):
    """Rolling attention probe (Max-of-Rolling-Means).

    REQUIRES SEQ axis.
    """

    def __init__(
        self,
        n_heads: int = 10,
        hidden_dim: int = 128,
        window_size: int = 10,
        dropout: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 20,
        patience: int = 5,
        batch_size: int = 32,
        val_split: float = 0.2,
        *,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
    ):
        super().__init__(device=device, seed=seed, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn, cast=cast)
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_split = val_split
        self.net = None

    @property
    def fitted(self) -> bool:
        return self.net is not None

    def _create_network(self, d_model: int) -> None:
        self.net = _RollingAttentionNetwork(
            d_model=d_model,
            n_heads=self.n_heads,
            hidden_dim=self.hidden_dim,
            window_size=self.window_size,
            dropout=self.dropout,
        )

    def fit(self, X: Activations, y: list | torch.Tensor) -> "RollingAttention":
        if "l" in X.dims:
            raise ValueError(
                f"RollingAttention expects no LAYER axis. "
                f'Call select("l", layer) first. Current dims: {X.dims}'
            )
        if "s" not in X.dims:
            raise ValueError("RollingAttention probe requires SEQ axis")

        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y)
        working_dtype = self._resolve_dtype(X.data.dtype)
        labels = y_tensor.to(self.device, dtype=working_dtype)

        g = self.setup_training(X.hidden_size, working_dtype)

        n_samples = X.batch_size
        n_val = max(1, int(self.val_split * n_samples))
        indices = torch.randperm(n_samples, device="cpu", generator=g)
        train_idx, val_idx = indices[n_val:].tolist(), indices[:n_val].tolist()
        batch_size = min(self.batch_size, len(train_idx))

        seq_lengths = self._get_seq_lengths(X)
        val_y = labels[val_idx]

        self.net.train()
        for epoch in range(self.n_epochs):
            batches = self._length_sorted_batches(
                train_idx, seq_lengths, batch_size, generator=g,
            )
            for batch_idx in batches:
                batch_seq, batch_mask = X.pad_batch(batch_idx)
                batch_seq = batch_seq.to(self.device)
                batch_mask = batch_mask.to(self.device)
                self.train_on_batch(batch_seq, batch_mask, labels[batch_idx])

            if self.should_validate_at(epoch):
                self.net.eval()
                with torch.no_grad():
                    val_logits = self._minibatch_forward(
                        self.net, X, val_idx, self.device,
                        working_dtype, batch_size,
                    )
                val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y).item()
                if self.check_val(val_loss):
                    break
                self.net.train()

        self.restore_best()
        self.net.eval()
        return self

    def __call__(
        self, sequences: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        self._check_fitted()
        sequences = sequences.to(self.device)
        mask = mask.to(self.device)
        return self.net(sequences, mask)

    def predict(self, X: Activations) -> torch.Tensor:
        self._check_fitted()
        if "l" in X.dims:
            raise ValueError(f"RollingAttention expects no LAYER axis. Current dims: {X.dims}")
        if "s" not in X.dims:
            raise ValueError("RollingAttention probe requires SEQ axis")

        net_dtype = next(self.net.parameters()).dtype
        all_indices = list(range(X.batch_size))

        with torch.no_grad():
            logits = self._minibatch_forward(
                self.net, X, all_indices, self.device,
                net_dtype, self.batch_size,
            )
            return torch.sigmoid(logits)

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "n_heads": self.n_heads,
            "hidden_dim": self.hidden_dim,
            "window_size": self.window_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "val_split": self.val_split,
            "seed": self.seed,
            "device": self.device,
            "cast": self.cast,
            "training_dtype": str(self._training_dtype),
            "network_state_dict": self.net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "RollingAttention":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            n_heads=state["n_heads"],
            hidden_dim=state["hidden_dim"],
            window_size=state["window_size"],
            dropout=state.get("dropout", 0.1),
            learning_rate=state.get("learning_rate", 5e-4),
            weight_decay=state.get("weight_decay", 1e-3),
            n_epochs=state["n_epochs"],
            patience=state["patience"],
            batch_size=state.get("batch_size", 32),
            val_split=state.get("val_split", 0.2),
            seed=state.get("seed"),
            device=device,
            cast=state.get("cast"),
        )
        dtype_str = state.get("training_dtype")
        stored_dtype = getattr(torch, dtype_str.split(".")[-1]) if dtype_str else torch.float32
        probe._training_dtype = stored_dtype

        d_model = state["network_state_dict"]["norm.weight"].shape[0]
        probe.net = _RollingAttentionNetwork(
            d_model=d_model,
            n_heads=probe.n_heads,
            hidden_dim=probe.hidden_dim,
            window_size=probe.window_size,
            dropout=probe.dropout,
        ).to(device, dtype=stored_dtype)
        probe.net.load_state_dict(state["network_state_dict"])
        probe.net.eval()
        return probe

    def __repr__(self) -> str:
        return f"RollingAttention(n_heads={self.n_heads}, window={self.window_size}, fitted={self.fitted})"
