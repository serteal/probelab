"""AlphaEvolve Gated Bipolar probe from GDM paper."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations
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
        self.net = None

    @property
    def fitted(self) -> bool:
        return self.net is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "GatedBipolar":
        if "l" in X.dims:
            raise ValueError(
                f"GatedBipolar expects no LAYER axis. "
                f"Call select_layers() first. Current dims: {X.dims}"
            )
        if "s" not in X.dims:
            raise ValueError("GatedBipolar probe requires SEQ axis")

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y)
        sequences = X.data.detach()
        mask = X.mask.detach()
        labels = y_tensor.float()
        d_model = sequences.shape[-1]

        # Fresh state every fit()
        self.net = _GatedBipolarNetwork(
            d_model=d_model,
            mlp_hidden_dim=self.mlp_hidden_dim,
            gate_dim=self.gate_dim,
            dropout=self.dropout,
        ).to(self.device)
        if sequences.dtype != torch.float32:
            self.net = self.net.to(sequences.dtype)

        optimizer = AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.startswith("cuda") if isinstance(self.device, str) else False,
        )

        # Train/validation split
        n_samples = len(sequences)
        n_val = max(1, int(0.2 * n_samples))
        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        batch_size = min(32, len(train_idx))

        best_val_loss = float("inf")
        patience_counter = 0

        self.net.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(train_idx))
            shuffled = train_idx[perm]

            for i in range(0, len(shuffled), batch_size):
                batch_idx = shuffled[i : i + batch_size]
                batch_seq = sequences[batch_idx].to(self.device)
                batch_mask = mask[batch_idx].to(self.device)
                batch_y = labels[batch_idx].to(self.device)

                optimizer.zero_grad()
                logits = self.net(batch_seq, batch_mask)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y) + \
                       self.net.get_regularization_loss(self.lambda_l1, self.lambda_orth)
                loss.backward()
                optimizer.step()

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_loss = F.binary_cross_entropy_with_logits(
                    self.net(sequences[val_idx].to(self.device), mask[val_idx].to(self.device)),
                    labels[val_idx].to(self.device)
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
            raise ValueError(f"GatedBipolar expects no LAYER axis. Current dims: {X.dims}")
        if "s" not in X.dims:
            raise ValueError("GatedBipolar probe requires SEQ axis")

        sequences = X.data.to(self.device)
        mask = X.mask.to(self.device)

        with torch.no_grad():
            return torch.sigmoid(self(sequences, mask))

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
            "network_state_dict": self.net.state_dict(),
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
        # Infer d_model from saved weights
        d_model = state["network_state_dict"]["norm.weight"].shape[0]
        probe.net = _GatedBipolarNetwork(
            d_model=d_model,
            mlp_hidden_dim=probe.mlp_hidden_dim,
            gate_dim=probe.gate_dim,
            dropout=probe.dropout,
        ).to(probe.device)
        probe.net.load_state_dict(state["network_state_dict"])
        probe.net.eval()
        return probe

    def __repr__(self) -> str:
        return f"GatedBipolar(gate_dim={self.gate_dim}, fitted={self.fitted})"
