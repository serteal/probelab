"""Multi-layer perceptron probe."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations, Axis
from ..processing.scores import Scores
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
        self._network = None
        self._optimizer = None
        self._d_model = None
        self._trained_on_tokens = False

    def _init_network(self, d_model: int):
        self._d_model = d_model
        self._network = _MLPNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device).to(torch.float32)
        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X: Activations, y: list | torch.Tensor) -> "MLP":
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                f"MLP expects no LAYER axis. Add SelectLayer({X.layer_indices[0]}) to pipeline."
            )

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.activations.device)

        y_tensor = self._to_labels(y)

        if X.has_axis(Axis.SEQ):
            features, tokens_per_sample = X.extract_tokens()
            if y_tensor.ndim == 1:
                labels = torch.repeat_interleave(y_tensor, tokens_per_sample.to(y_tensor.device))
            elif y_tensor.ndim == 2:
                labels = y_tensor[X.detection_mask.cpu().bool()]
            else:
                raise ValueError(f"Invalid label shape: {y_tensor.shape}")
            self._trained_on_tokens = True
            self._tokens_per_sample = tokens_per_sample
        else:
            features = X.activations
            labels = y_tensor
            self._trained_on_tokens = False
            self._tokens_per_sample = None

        if features.shape[0] == 0:
            return self

        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device).float()

        if self._network is None:
            self._init_network(features.shape[1])

        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self._network.train()
        for _ in range(self.n_epochs):
            for batch_features, batch_labels in dataloader:
                self._optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    self._network(batch_features), batch_labels
                )
                loss.backward()
                self._optimizer.step()

        self._network.eval()
        self._fitted = True
        return self

    def predict(self, X: Activations) -> Scores:
        self._check_fitted()

        if X.has_axis(Axis.LAYER):
            raise ValueError("MLP expects no LAYER axis")

        if X.has_axis(Axis.SEQ):
            features, tokens_per_sample = X.extract_tokens()
            is_token_level = True
        else:
            features = X.activations
            tokens_per_sample = None
            is_token_level = False

        features = features.to(self.device, dtype=torch.float32)

        self._network.eval()
        with torch.no_grad():
            probs_pos = torch.sigmoid(self._network(features))
            probs = torch.stack([1 - probs_pos, probs_pos], dim=-1)

        if is_token_level:
            return Scores.from_token_scores(probs, tokens_per_sample, X.batch_indices)
        return Scores.from_sequence_scores(probs, X.batch_indices)

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
            "d_model": self._d_model,
            "network_state": self._network.state_dict(),
            "trained_on_tokens": self._trained_on_tokens,
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
        probe._d_model = state["d_model"]
        probe._init_network(probe._d_model)
        probe._network.load_state_dict(state["network_state"])
        probe._network.eval()
        probe._trained_on_tokens = state.get("trained_on_tokens", False)
        probe._fitted = True
        return probe

    def __repr__(self) -> str:
        token_str = ", token-level" if self._trained_on_tokens else ""
        return f"MLP(hidden_dim={self.hidden_dim}, fitted={self._fitted}{token_str})"
