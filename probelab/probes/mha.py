"""Multi-head self-attention (transformer encoder) probe.

Down-projects tokens, prepends learnable CLS token, runs through
a small transformer encoder, classifies from CLS output.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import Activations
from .base import BaseProbe


class _MHANetwork(nn.Module):
    """Lightweight transformer encoder with CLS token."""

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 128,
        n_heads: int = 4,
        n_enc_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.down_proj = nn.Linear(d_model, proj_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=4 * proj_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_enc_layers, enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(proj_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = sequences.shape[0]
        x = self.down_proj(sequences)  # [B, S, proj_dim]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, proj_dim]
        x = torch.cat([cls, x], dim=1)  # [B, S+1, proj_dim]

        # Extend mask for CLS (always valid)
        cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
        ext_mask = torch.cat([cls_mask, mask], dim=1)  # [B, S+1]

        # TransformerEncoder expects src_key_padding_mask: True = IGNORE
        x = self.encoder(x, src_key_padding_mask=~ext_mask.bool())

        # Classify from CLS output
        return self.classifier(x[:, 0]).squeeze(-1)  # [B]


class MHA(BaseProbe):
    """Multi-head self-attention (transformer encoder) probe.

    REQUIRES SEQ axis.
    """

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
        self.net = None

    @property
    def fitted(self) -> bool:
        return self.net is not None

    def _create_network(self, d_model: int) -> None:
        self.net = _MHANetwork(
            d_model=d_model,
            proj_dim=self.proj_dim,
            n_heads=self.n_heads,
            n_enc_layers=self.n_enc_layers,
            dropout=self.dropout,
        )

    def fit(self, X: Activations, y: list | torch.Tensor) -> "MHA":
        if "l" in X.dims:
            raise ValueError(
                f"MHA expects no LAYER axis. "
                f'Call select("l", layer) first. Current dims: {X.dims}'
            )
        if "s" not in X.dims:
            raise ValueError("MHA probe requires SEQ axis")

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
            raise ValueError(f"MHA expects no LAYER axis. Current dims: {X.dims}")
        if "s" not in X.dims:
            raise ValueError("MHA probe requires SEQ axis")

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
            "seed": self.seed,
            "device": self.device,
            "cast": self.cast,
            "training_dtype": str(self._training_dtype),
            "network_state_dict": self.net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MHA":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            proj_dim=state["proj_dim"],
            n_heads=state["n_heads"],
            n_enc_layers=state["n_enc_layers"],
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

        d_model = state["network_state_dict"]["down_proj.weight"].shape[1]
        probe.net = _MHANetwork(
            d_model=d_model,
            proj_dim=probe.proj_dim,
            n_heads=probe.n_heads,
            n_enc_layers=probe.n_enc_layers,
            dropout=probe.dropout,
        ).to(device, dtype=stored_dtype)
        probe.net.load_state_dict(state["network_state_dict"])
        probe.net.eval()
        return probe

    def __repr__(self) -> str:
        return f"MHA(proj_dim={self.proj_dim}, n_heads={self.n_heads}, fitted={self.fitted})"
