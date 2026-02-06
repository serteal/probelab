"""Multi-head hard max pooling probe from GDM paper."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations
from .base import BaseProbe


class _MultiMaxNetwork(nn.Module):
    """Multi-head hard max pooling network."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 10,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )
        self.head_projections = nn.ModuleList(
            [nn.Linear(mlp_hidden_dim, 1) for _ in range(n_heads)]
        )
        self.output = nn.Linear(n_heads, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(self.norm(sequences))
        head_outputs = []
        for head_proj in self.head_projections:
            scores = head_proj(mlp_out).squeeze(-1)
            scores_masked = scores.masked_fill(~mask.bool(), float("-inf"))
            head_outputs.append(scores_masked.max(dim=1).values)
        head_stack = torch.nan_to_num(torch.stack(head_outputs, dim=1), nan=0.0, posinf=0.0, neginf=0.0)
        return self.output(head_stack).squeeze(-1)


class MultiMax(BaseProbe):
    """Multi-head hard max pooling probe.

    Uses H independent heads, each selecting the highest-scoring token via hard max.
    REQUIRES SEQ axis.
    """

    def __init__(
        self,
        n_heads: int = 10,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 20,
        patience: int = 5,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.n_heads = n_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self._network = None
        self._optimizer = None
        self._d_model = None

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        self._d_model = d_model
        self._network = _MultiMaxNetwork(
            d_model=d_model,
            n_heads=self.n_heads,
            mlp_hidden_dim=self.mlp_hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        if dtype is not None:
            self._network = self._network.to(dtype)
        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.startswith("cuda") if isinstance(self.device, str) else False,
        )

    def fit(self, X: Activations, y: list | torch.Tensor) -> "MultiMax":
        if "l" in X.dims:
            raise ValueError(
                f"MultiMax expects no LAYER axis. "
                f"Call select_layers() first. Current dims: {X.dims}"
            )
        if "s" not in X.dims:
            raise ValueError("MultiMax probe requires SEQ axis")

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y)
        sequences = X.data.detach()
        mask = X.mask.detach()
        labels = y_tensor.float()

        if self._network is None:
            self._init_network(sequences.shape[-1], dtype=sequences.dtype)

        # Train/validation split
        n_samples = len(sequences)
        n_val = max(1, int(0.2 * n_samples))
        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        batch_size = min(32, len(train_idx))

        best_val_loss = float("inf")
        patience_counter = 0

        self._network.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(train_idx))
            shuffled = train_idx[perm]

            for i in range(0, len(shuffled), batch_size):
                batch_idx = shuffled[i : i + batch_size]
                batch_seq = sequences[batch_idx].to(self.device)
                batch_mask = mask[batch_idx].to(self.device)
                batch_y = labels[batch_idx].to(self.device)

                self._optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    self._network(batch_seq, batch_mask), batch_y
                )
                loss.backward()
                self._optimizer.step()

            # Validation
            self._network.eval()
            with torch.no_grad():
                val_loss = F.binary_cross_entropy_with_logits(
                    self._network(sequences[val_idx].to(self.device), mask[val_idx].to(self.device)),
                    labels[val_idx].to(self.device)
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

    def __call__(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass.

        Args:
            sequences: Sequence tensor [batch, seq, hidden]
            mask: Detection mask [batch, seq]

        Returns:
            Logits tensor [batch]
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

        if "l" in X.dims:
            raise ValueError(f"MultiMax expects no LAYER axis. Current dims: {X.dims}")
        if "s" not in X.dims:
            raise ValueError("MultiMax probe requires SEQ axis")

        sequences = X.data.to(self.device)
        mask = X.mask.to(self.device)

        with torch.no_grad():
            return torch.sigmoid(self(sequences, mask))

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "n_heads": self.n_heads,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "device": self.device,
            "d_model": self._d_model,
            "network_state_dict": self._network.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MultiMax":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            n_heads=state["n_heads"],
            mlp_hidden_dim=state["mlp_hidden_dim"],
            dropout=state.get("dropout", 0.1),
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
        return f"MultiMax(n_heads={self.n_heads}, fitted={self._fitted})"
