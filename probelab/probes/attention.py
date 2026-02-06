"""Attention-based probe."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations, Axis
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
        self._network = None
        self._optimizer = None
        self._d_model = None
        self.attention_weights = None

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        self._d_model = d_model
        self._network = _AttentionNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            temperature=self.temperature,
        ).to(self.device)
        if dtype is not None:
            self._network = self._network.to(dtype)
        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.startswith("cuda"),
        )

    def fit(self, X: Activations, y: list | torch.Tensor) -> "Attention":
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                f"Attention expects no LAYER axis. Add SelectLayer({X.layer_indices[0]}) to pipeline."
            )
        if not X.has_axis(Axis.SEQ):
            raise ValueError("Attention probe requires SEQ axis")

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.activations.device)

        y_tensor = self._to_labels(y)
        sequences = X.activations.clone().to(self.device)
        detection_mask = X.detection_mask.to(self.device)
        labels = y_tensor.to(self.device).float()

        if self._network is None:
            self._init_network(sequences.shape[-1], dtype=sequences.dtype)

        # Train/validation split
        n_samples = len(sequences)
        n_val = max(1, int(0.2 * n_samples))
        indices = torch.randperm(n_samples, device=self.device)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_seq, train_mask, train_y = sequences[train_idx], detection_mask[train_idx], labels[train_idx]
        val_seq, val_mask, val_y = sequences[val_idx], detection_mask[val_idx], labels[val_idx]

        best_val_loss = float("inf")
        patience_counter = 0

        self._network.train()
        for epoch in range(self.n_epochs):
            self._optimizer.zero_grad()
            logits, _ = self._network(train_seq, train_mask)
            F.binary_cross_entropy_with_logits(logits, train_y).backward()
            self._optimizer.step()

            if epoch % 10 == 0:
                self._network.eval()
                with torch.no_grad():
                    val_loss = F.binary_cross_entropy_with_logits(
                        self._network(val_seq, val_mask)[0], val_y
                    )
                self._network.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
                if val_loss < 0.001:
                    break

        self._network.eval()
        self._fitted = True
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
        return self._network(sequences, mask)

    def predict(self, X: Activations) -> torch.Tensor:
        """Evaluate on activations.

        Args:
            X: Activations with SEQ axis, without LAYER axis

        Returns:
            Probability of positive class [batch]
        """
        self._check_fitted()

        if X.has_axis(Axis.LAYER):
            raise ValueError("Attention expects no LAYER axis")
        if not X.has_axis(Axis.SEQ):
            raise ValueError("Attention probe requires SEQ axis")

        sequences = X.activations.to(self.device)
        detection_mask = X.detection_mask.to(self.device)

        with torch.no_grad():
            logits, attn_weights = self(sequences, detection_mask)
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
            "device": self.device,
            "d_model": self._d_model,
            "network_state_dict": self._network.state_dict(),
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
            device=device,
        )
        probe._d_model = state["d_model"]
        probe._init_network(probe._d_model)
        probe._network.load_state_dict(state["network_state_dict"])
        probe._network.eval()
        probe._fitted = True
        return probe

    def __repr__(self) -> str:
        return f"Attention(hidden_dim={self.hidden_dim}, fitted={self._fitted})"
