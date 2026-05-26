"""Positional attention probe.

Cross-attention with learned query tokens per head and ALiBi-style
position bias. Lightweight — no MLP on activations, operates directly
on raw hidden states.

Reference: EleutherAI blog "Attention Probes" (2025);
           McKenzie et al., "Detecting High-Stakes Interactions" (2025).
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class _PositionalAttentionNetwork(nn.Module):
    """Multi-head cross-attention with ALiBi position bias."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        # Per-head query and value projections: [d_model, n_heads]
        self.query_proj = nn.Parameter(torch.zeros(d_model, n_heads))
        self.value_proj = nn.Parameter(torch.empty(d_model, n_heads))
        nn.init.xavier_normal_(self.value_proj, gain=0.5)
        # Learned ALiBi-style position bias slopes
        self.position_weights = nn.Parameter(torch.zeros(n_heads))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = sequences.shape
        # Attention logits: [B, S, H]
        attn_logits = sequences @ self.query_proj

        # Position bias: [S, H]
        positions = torch.arange(S, device=sequences.device, dtype=sequences.dtype)
        pos_bias = positions.unsqueeze(-1) * self.position_weights.unsqueeze(0)
        attn_logits = attn_logits + pos_bias.unsqueeze(0)

        # Mask invalid positions
        mask_bool = mask.bool()
        attn_logits = attn_logits.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))

        # Softmax over sequence dim
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Values: [B, S, H]
        values = sequences @ self.value_proj

        # Weighted sum over sequence and heads → scalar
        output = (attn_weights * values).sum(dim=(1, 2)) + self.bias
        return output


class PositionalAttention(BaseProbe):
    """Positional attention probe with ALiBi position bias.

    Lightweight cross-attention applied directly to raw activations (no MLP).
    Per-head learned query and value projections with position-dependent bias.
    REQUIRES SEQ axis.
    """

    def __init__(
        self,
        n_heads: int = 8,
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
        self.net = _PositionalAttentionNetwork(d_model=d_model, n_heads=self.n_heads)

    def fit(self, X: Activations, y: list | torch.Tensor) -> "PositionalAttention":
        if "l" in X.dims:
            raise ValueError(
                f"PositionalAttention expects no LAYER axis. "
                f'Call select("l", layer) first. Current dims: {X.dims}'
            )
        if "s" not in X.dims:
            raise ValueError("PositionalAttention probe requires SEQ axis")

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
            raise ValueError(f"PositionalAttention expects no LAYER axis. Current dims: {X.dims}")
        if "s" not in X.dims:
            raise ValueError("PositionalAttention probe requires SEQ axis")

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
    def load(cls, path: Path | str, device: str = "cpu") -> "PositionalAttention":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            n_heads=state["n_heads"],
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

        d_model = state["network_state_dict"]["query_proj"].shape[0]
        probe.net = _PositionalAttentionNetwork(
            d_model=d_model, n_heads=probe.n_heads,
        ).to(device, dtype=stored_dtype)
        probe.net.load_state_dict(state["network_state_dict"])
        probe.net.eval()
        return probe

    def __repr__(self) -> str:
        return f"PositionalAttention(n_heads={self.n_heads}, fitted={self.fitted})"
