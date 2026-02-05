"""AlphaEvolve Gated Bipolar probe from GDM paper."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations, Axis
from ..processing.scores import Scores
from .base import BaseProbe


class _GatedBipolarNetwork(nn.Module):
    """Gated Bipolar network with AlphaEvolve architecture."""

    def __init__(
        self,
        d_model: int,
        mlp_hidden_dim: int = 128,
        gate_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate_dim = gate_dim
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )
        self.W_proj = nn.Linear(mlp_hidden_dim, gate_dim)
        self.W_gate = nn.Linear(mlp_hidden_dim, gate_dim)
        self.output = nn.Linear(2 * gate_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        H = self.mlp(self.norm(sequences))
        V = self.W_proj(H) * F.softplus(self.W_gate(H))
        mask_exp = mask.unsqueeze(-1)

        V_max = V.masked_fill(~mask_exp.bool(), float("-inf")).max(dim=1).values
        V_min = V.masked_fill(~mask_exp.bool(), float("inf")).min(dim=1).values
        V_max = torch.nan_to_num(V_max, nan=0.0, posinf=0.0, neginf=0.0)
        V_min = torch.nan_to_num(V_min, nan=0.0, posinf=0.0, neginf=0.0)

        h_pool = torch.cat([V_max, -V_min], dim=-1)
        return self.output(h_pool).squeeze(-1)

    def get_regularization_loss(self, lambda_l1: float = 1e-5, lambda_orth: float = 1e-4) -> torch.Tensor:
        l1_loss = lambda_l1 * self.W_proj.weight.abs().sum()
        W = self.W_proj.weight
        WtW = W @ W.T
        I = torch.eye(self.gate_dim, device=W.device, dtype=W.dtype)
        orth_loss = lambda_orth * (WtW - I).pow(2).sum()
        return l1_loss + orth_loss


class GatedBipolar(BaseProbe):
    """AlphaEvolve Gated Bipolar probe.

    Uses gated projections with Softplus and bipolar pooling (max AND -min).
    REQUIRES SEQ axis.
    """

    def __init__(
        self,
        mlp_hidden_dim: int = 128,
        gate_dim: int = 64,
        dropout: float = 0.1,
        lambda_l1: float = 1e-5,
        lambda_orth: float = 1e-4,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        n_epochs: int = 20,
        patience: int = 5,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.gate_dim = gate_dim
        self.dropout = dropout
        self.lambda_l1 = lambda_l1
        self.lambda_orth = lambda_orth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self._network = None
        self._optimizer = None
        self._d_model = None

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        self._d_model = d_model
        self._network = _GatedBipolarNetwork(
            d_model=d_model,
            mlp_hidden_dim=self.mlp_hidden_dim,
            gate_dim=self.gate_dim,
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

    def fit(self, X: Activations, y: list | torch.Tensor) -> "GatedBipolar":
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                f"GatedBipolar expects no LAYER axis. Add SelectLayer({X.layer_indices[0]}) to pipeline."
            )
        if not X.has_axis(Axis.SEQ):
            raise ValueError("GatedBipolar probe requires SEQ axis")

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.activations.device)

        y_tensor = self._to_labels(y)
        sequences = X.activations.detach()
        detection_mask = X.detection_mask.detach()
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
                batch_mask = detection_mask[batch_idx].to(self.device)
                batch_y = labels[batch_idx].to(self.device)

                self._optimizer.zero_grad()
                logits = self._network(batch_seq, batch_mask)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y) + \
                       self._network.get_regularization_loss(self.lambda_l1, self.lambda_orth)
                loss.backward()
                self._optimizer.step()

            # Validation
            self._network.eval()
            with torch.no_grad():
                val_loss = F.binary_cross_entropy_with_logits(
                    self._network(sequences[val_idx].to(self.device), detection_mask[val_idx].to(self.device)),
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

    def predict(self, X: Activations) -> Scores:
        self._check_fitted()

        if X.has_axis(Axis.LAYER):
            raise ValueError("GatedBipolar expects no LAYER axis")
        if not X.has_axis(Axis.SEQ):
            raise ValueError("GatedBipolar probe requires SEQ axis")

        sequences = X.activations.to(self.device)
        detection_mask = X.detection_mask.to(self.device)

        self._network.eval()
        with torch.no_grad():
            probs_pos = torch.sigmoid(self._network(sequences, detection_mask))
            probs = torch.stack([1 - probs_pos, probs_pos], dim=-1)

        return Scores.from_sequence_scores(probs, X.batch_indices)

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "gate_dim": self.gate_dim,
            "dropout": self.dropout,
            "lambda_l1": self.lambda_l1,
            "lambda_orth": self.lambda_orth,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "device": self.device,
            "d_model": self._d_model,
            "network_state_dict": self._network.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "GatedBipolar":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            mlp_hidden_dim=state["mlp_hidden_dim"],
            gate_dim=state["gate_dim"],
            dropout=state.get("dropout", 0.1),
            lambda_l1=state.get("lambda_l1", 1e-5),
            lambda_orth=state.get("lambda_orth", 1e-4),
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
        return f"GatedBipolar(gate_dim={self.gate_dim}, fitted={self._fitted})"
