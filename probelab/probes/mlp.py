"""Multi-layer perceptron probe."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations
from .base import BaseProbe


class _MLPNetwork(nn.Module):
    """Simple MLP architecture for binary classification."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float | None = None,
        activation: Literal["relu", "gelu"] = "relu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class MLP(BaseProbe):
    """Multi-layer perceptron probe.

    Adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float | None = None,
        activation: Literal["relu", "gelu"] = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.net = None

    @property
    def fitted(self) -> bool:
        return self.net is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "MLP":
        if "l" in X.dims:
            raise ValueError(
                f"MLP expects no LAYER axis. "
                f"Call select_layers() first. Current dims: {X.dims}"
            )

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y)

        if "s" in X.dims:
            features, tokens_per_sample = X.extract_tokens()
            if y_tensor.ndim == 1:
                labels = torch.repeat_interleave(y_tensor, tokens_per_sample.to(y_tensor.device))
            elif y_tensor.ndim == 2:
                # 2D token-level labels: extract via det mask
                det_bool = X.det.bool()
                labels = y_tensor.view(-1)[det_bool[:y_tensor.numel()]] if y_tensor.numel() > 0 else y_tensor.new_empty(0)
            else:
                raise ValueError(f"Invalid label shape: {y_tensor.shape}")
        else:
            features = X.data
            labels = y_tensor

        if features.shape[0] == 0:
            return self

        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device).float()
        d_model = features.shape[1]

        # Fresh state every fit()
        self.net = _MLPNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device).to(torch.float32)

        optimizer = AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.net.train()
        for _ in range(self.n_epochs):
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    self.net(batch_features), batch_labels
                )
                loss.backward()
                optimizer.step()

        self.net.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass.

        Args:
            x: Features tensor [batch, hidden] or [n_tokens, hidden]

        Returns:
            Logits tensor [batch] or [n_tokens] (not probabilities)
        """
        self._check_fitted()
        x = x.to(self.device, dtype=torch.float32)
        return self.net(x)

    def predict(self, X: Activations) -> torch.Tensor:
        """Evaluate on activations.

        Args:
            X: Activations without LAYER axis

        Returns:
            Probability of positive class:
            - [batch, seq] if X has SEQ axis (token-level)
            - [batch] if X has no SEQ axis (sequence-level)
        """
        self._check_fitted()

        if "l" in X.dims:
            raise ValueError(f"MLP expects no LAYER axis. Current dims: {X.dims}")

        with torch.no_grad():
            if "s" in X.dims:
                features, _ = X.extract_tokens()
                flat_probs = torch.sigmoid(self(features))

                # Scatter back to [batch, seq] via to_padded()
                padded_data, padded_det = X.to_padded()
                probs = torch.zeros_like(padded_det, dtype=flat_probs.dtype)
                probs[padded_det] = flat_probs
                return probs
            else:
                return torch.sigmoid(self(X.data))

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "network_state": self.net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MLP":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            hidden_dim=state["hidden_dim"],
            dropout=state.get("dropout"),
            activation=state["activation"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            batch_size=state["batch_size"],
            device=device,
        )
        # Infer d_model from saved weights
        d_model = state["network_state"]["fc1.weight"].shape[1]
        probe.net = _MLPNetwork(
            d_model=d_model,
            hidden_dim=probe.hidden_dim,
            dropout=probe.dropout,
            activation=probe.activation,
        ).to(probe.device).to(torch.float32)
        probe.net.load_state_dict(state["network_state"])
        probe.net.eval()
        return probe

    def __repr__(self) -> str:
        return f"MLP(hidden_dim={self.hidden_dim}, fitted={self.fitted})"
