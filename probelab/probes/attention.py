"""Attention-based probe."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations
from .base import BaseProbe


class _AttentionNetwork(nn.Module):
    """Attention-based neural network for sequence classification."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, sequences: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.attention_norm(sequences)
        scores = self.attention_scorer(normed).squeeze(-1) / self.temperature
        scores_masked = scores.masked_fill(~mask.bool(), float("-inf"))
        attn_weights = torch.nan_to_num(torch.softmax(scores_masked, dim=1), nan=0.0)
        aggregated = (attn_weights.unsqueeze(-1) * sequences).sum(dim=1)
        logits = self.classifier(self.classifier_norm(aggregated)).squeeze(-1)
        return logits, attn_weights


class Attention(BaseProbe):
    """Attention-based probe for sequence classification.

    Learns attention weights over the sequence dimension. REQUIRES SEQ axis.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        temperature: float = 2.0,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 1000,
        patience: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.net = None
        self.attention_weights = None

    @property
    def fitted(self) -> bool:
        return self.net is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "Attention":
        if "l" in X.dims:
            raise ValueError(
                f"Attention expects no LAYER axis. "
                f"Call select_layers() first. Current dims: {X.dims}"
            )
        if "s" not in X.dims:
            raise ValueError("Attention probe requires SEQ axis")

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y)
        labels = y_tensor.to(self.device).float()
        d_model = X.hidden_size

        # Fresh state every fit()
        self.net = _AttentionNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            temperature=self.temperature,
        ).to(self.device)

        optimizer = AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.startswith("cuda") if isinstance(self.device, str) else False,
        )

        # Train/validation split
        n_samples = X.batch_size
        n_val = max(1, int(0.2 * n_samples))
        indices = torch.randperm(n_samples, device="cpu")
        train_idx, val_idx = indices[n_val:].tolist(), indices[:n_val].tolist()
        batch_size = min(self.batch_size, len(train_idx))

        # Pre-pad validation set
        val_seq, val_mask = X.pad_batch(val_idx)
        val_seq = val_seq.to(self.device)
        val_mask = val_mask.to(self.device)
        val_y = labels[val_idx]

        # Check dtype
        if val_seq.dtype != torch.float32:
            self.net = self.net.to(val_seq.dtype)

        best_val_loss = float("inf")
        patience_counter = 0

        self.net.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(train_idx))
            shuffled = [train_idx[p] for p in perm.tolist()]

            for i in range(0, len(shuffled), batch_size):
                batch_idx = shuffled[i : i + batch_size]
                batch_seq, batch_mask = X.pad_batch(batch_idx)
                batch_seq = batch_seq.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_y = labels[batch_idx]

                optimizer.zero_grad()
                logits, _ = self.net(batch_seq, batch_mask)
                F.binary_cross_entropy_with_logits(logits, batch_y).backward()
                optimizer.step()

            if epoch % 10 == 0:
                self.net.eval()
                with torch.no_grad():
                    val_loss = F.binary_cross_entropy_with_logits(
                        self.net(val_seq, val_mask)[0], val_y
                    )
                self.net.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
                if val_loss < 0.001:
                    break

        self.net.eval()
        return self

    def __call__(
        self, sequences: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable forward pass.

        Args:
            sequences: Sequence tensor [batch, seq, hidden]
            mask: Detection mask [batch, seq]

        Returns:
            Tuple of (logits [batch], attention_weights [batch, seq])
        """
        self._check_fitted()
        sequences = sequences.to(self.device)
        mask = mask.to(self.device)
        return self.net(sequences, mask)

    def predict(self, X: Activations) -> torch.Tensor:
        """Evaluate on activations.

        Args:
            X: Activations with SEQ axis, without LAYER axis

        Returns:
            Probability of positive class [batch]
        """
        self._check_fitted()

        if "l" in X.dims:
            raise ValueError(f"Attention expects no LAYER axis. Current dims: {X.dims}")
        if "s" not in X.dims:
            raise ValueError("Attention probe requires SEQ axis")

        sequences, mask = X.to_padded()
        sequences = sequences.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            logits, attn_weights = self(sequences, mask)
            self.attention_weights = attn_weights.detach().cpu()
            return torch.sigmoid(logits)

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "device": self.device,
            "network_state_dict": self.net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Attention":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            hidden_dim=state["hidden_dim"],
            dropout=state.get("dropout", 0.2),
            temperature=state.get("temperature", 2.0),
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            patience=state["patience"],
            batch_size=state.get("batch_size", 32),
            device=device,
        )
        # Infer d_model from saved weights
        d_model = state["network_state_dict"]["attention_norm.weight"].shape[0]
        probe.net = _AttentionNetwork(
            d_model=d_model,
            hidden_dim=probe.hidden_dim,
            dropout=probe.dropout,
            temperature=probe.temperature,
        ).to(probe.device)
        probe.net.load_state_dict(state["network_state_dict"])
        probe.net.eval()
        return probe

    def __repr__(self) -> str:
        return f"Attention(hidden_dim={self.hidden_dim}, fitted={self.fitted})"
