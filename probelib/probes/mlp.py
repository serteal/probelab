"""
Multi-layer perceptron probe implementation following sklearn-style API.
"""

from pathlib import Path
from typing import Literal, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import ActivationIterator, Activations
from .base import BaseProbe


class MLPNetwork(nn.Module):
    """
    Simple MLP architecture for binary classification.

    Architecture: input -> hidden -> ReLU/GELU -> dropout (optional) -> output
    """

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
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # Return logits [batch_size]


class MLP(BaseProbe):
    """
    Multi-layer perceptron probe for binary classification.

    This probe uses a simple MLP with one hidden layer. It supports
    both batch and streaming training modes via AdamW optimizer.

    Attributes:
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability (None for no dropout)
        activation: Activation function ("relu" or "gelu")
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: L2 regularization strength
        n_epochs: Maximum number of training epochs
        patience: Early stopping patience
        _network: MLPNetwork instance
        _optimizer: AdamW optimizer (persisted for streaming)
        _d_model: Input dimension (set during first fit)
    """

    def __init__(
        self,
        layer: int,
        sequence_aggregation: Literal["mean", "max", "last_token"] | None = None,
        score_aggregation: Literal["mean", "max", "last_token"] | None = None,
        hidden_dim: int = 128,
        dropout: float | None = None,
        activation: Literal["relu", "gelu"] = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize MLP probe.

        Args:
            layer: Layer index to use activations from
            sequence_aggregation: Aggregate sequences BEFORE training (classic)
            score_aggregation: Train on tokens, aggregate AFTER prediction
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability (None for no dropout)
            activation: Activation function ("relu" or "gelu")
            learning_rate: Learning rate for AdamW
            weight_decay: Weight decay for AdamW (L2 regularization)
            n_epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device for PyTorch operations
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        # Default to sequence_aggregation="mean" if neither is specified
        if sequence_aggregation is None and score_aggregation is None:
            sequence_aggregation = "mean"

        super().__init__(
            layer=layer,
            sequence_aggregation=sequence_aggregation,
            score_aggregation=score_aggregation,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._d_model = None
        self._streaming_steps = 0

        # This probe requires gradients for training
        self._requires_grad = True

        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

    def _init_network(self, d_model: int):
        """Initialize the network once we know the input dimension."""
        self._d_model = d_model
        self._network = MLPNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.startswith("cuda"),
        )

    def fit(self, X: Activations | ActivationIterator, y: list | torch.Tensor) -> Self:
        """
        Fit the probe on training data.

        Args:
            X: Activations or ActivationIterator containing features
            y: Labels for training

        Returns:
            self: Fitted probe instance
        """
        if isinstance(X, ActivationIterator):
            # Use streaming approach for iterators
            return self._fit_iterator(X, y)

        # Standard batch fitting
        features = self._prepare_features(X)
        labels = self._prepare_labels(
            y, expand_for_tokens=(self.sequence_aggregation is None)
        )

        # Move to device and ensure float32 for MLP compatibility
        features = features.to(self.device).float()
        labels = labels.to(self.device).float()

        # Validate we have both classes
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Training data must contain both classes. Found: {unique_labels.tolist()}"
            )

        # Initialize network if needed
        if self._network is None:
            self._init_network(features.shape[1])

        # Create train/validation split
        n_samples = len(features)
        n_val = max(1, int(0.2 * n_samples))  # At least 1 validation sample

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        indices = torch.randperm(n_samples, device=self.device)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        X_train = features[train_indices]
        y_train = labels[train_indices]
        X_val = features[val_indices]
        y_val = labels[val_indices]

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        self._network.train()
        for epoch in range(self.n_epochs):
            # Training step
            self._optimizer.zero_grad()
            logits = self._network(X_train)
            loss = F.binary_cross_entropy_with_logits(logits, y_train)
            loss.backward()
            self._optimizer.step()

            # Validation step
            self._network.eval()
            with torch.no_grad():
                val_logits = self._network(X_val)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val)
            self._network.train()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            # Stop if loss is very small
            if val_loss < 0.01:
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                break

        self._network.eval()
        self._fitted = True
        return self

    def _fit_iterator(self, X: ActivationIterator, y: list | torch.Tensor) -> Self:
        """
        Fit using an ActivationIterator (streaming mode).

        Args:
            X: ActivationIterator yielding batches of activations
            y: All labels

        Returns:
            self: Fitted probe instance
        """
        labels_tensor = self._prepare_labels(y)

        # Process batches
        for batch_acts in X:
            batch_idx = torch.tensor(
                batch_acts.batch_indices, device=labels_tensor.device, dtype=torch.long
            )
            batch_labels = labels_tensor.index_select(0, batch_idx)
            self.partial_fit(batch_acts, batch_labels)

        return self

    def partial_fit(self, X: Activations, y: list | torch.Tensor) -> Self:
        """
        Incrementally fit the probe on a batch of samples.

        Args:
            X: Activations containing features for this batch
            y: Labels for this batch

        Returns:
            self: Partially fitted probe instance
        """
        features = self._prepare_features(X)
        labels = self._prepare_labels(
            y, expand_for_tokens=(self.sequence_aggregation is None)
        )

        # Move to device and ensure float32 for MLP compatibility
        features = features.to(self.device).float()
        labels = labels.to(self.device).float()

        # Initialize network on first batch
        if self._network is None:
            self._init_network(features.shape[1])

        # Training step
        self._network.train()
        self._optimizer.zero_grad()
        logits = self._network(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self._optimizer.step()

        self._streaming_steps += 1
        self._fitted = True

        # Switch back to eval mode
        self._network.eval()
        return self

    def predict_proba(self, X: Activations | ActivationIterator) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            X: Activations or ActivationIterator containing features

        Returns:
            Tensor of shape (n_samples, 2) with probabilities for each class
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        if isinstance(X, ActivationIterator):
            # Predict on iterator batches
            return self._predict_iterator(X)

        # Standard prediction - always aggregate for prediction
        # Get features based on training mode
        features = self._prepare_features(X)
        features = features.to(self.device).float()  # Ensure float32 for MLP

        # Get predictions
        self._network.eval()
        with torch.no_grad():
            logits = self._network(features)

            # Apply score aggregation if needed (aggregate logits, not probabilities)
            if self.score_aggregation is not None and hasattr(
                self, "_tokens_per_sample"
            ):
                # Aggregate logits before applying sigmoid
                logits = self._aggregate_scores(
                    logits.unsqueeze(-1), self.score_aggregation
                ).squeeze(-1)

            probs_positive = torch.sigmoid(logits)

        # Create 2-class probability matrix
        n_predictions = len(probs_positive)
        probs = torch.zeros(n_predictions, 2, device=self.device)
        probs[:, 0] = 1 - probs_positive  # P(y=0)
        probs[:, 1] = probs_positive  # P(y=1)

        return probs

    def _predict_iterator(self, X: ActivationIterator) -> torch.Tensor:
        """
        Predict on an ActivationIterator.

        Args:
            X: ActivationIterator yielding batches

        Returns:
            Concatenated predictions for all batches
        """
        all_probs = []

        for batch_acts in X:
            batch_probs = self.predict_proba(batch_acts)
            all_probs.append(batch_probs)

        return torch.cat(all_probs, dim=0)

    def save(self, path: Path | str) -> None:
        """
        Save the probe to disk.

        Args:
            path: Path to save the probe
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted probe")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        state = {
            "layer": self.layer,
            "sequence_aggregation": self.sequence_aggregation,
            "score_aggregation": self.score_aggregation,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "d_model": self._d_model,
            "network_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "streaming_steps": self._streaming_steps,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> Self:
        """
        Load a probe from disk.

        Args:
            path: Path to load the probe from
            device: Device to load onto (None to use saved device)

        Returns:
            Loaded probe instance
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu")

        # Create probe instance
        probe = cls(
            layer=state["layer"],
            sequence_aggregation=state["sequence_aggregation"],
            score_aggregation=state["score_aggregation"],
            hidden_dim=state["hidden_dim"],
            dropout=state["dropout"],
            activation=state["activation"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            patience=state["patience"],
            device=device or state["device"],
            random_state=state["random_state"],
            verbose=state.get("verbose", False),
        )

        # Initialize network
        probe._init_network(state["d_model"])

        # Load network and optimizer states
        probe._network.load_state_dict(state["network_state_dict"])
        probe._optimizer.load_state_dict(state["optimizer_state_dict"])
        probe._streaming_steps = state["streaming_steps"]

        # Move to correct device
        probe._network = probe._network.to(probe.device)
        probe._network.eval()
        probe._fitted = True

        return probe
