"""Multi-layer perceptron probe implementation."""

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
    """Simple MLP architecture for binary classification (internal implementation).

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
    """Multi-layer perceptron probe for binary classification.

    This probe adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores

    Uses AdamW optimizer for training over multiple epochs.

    Args:
        hidden_dim: Number of hidden units in the MLP
        dropout: Dropout rate (None for no dropout)
        activation: Activation function ("relu" or "gelu")
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength
        n_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        device: Device for computation (auto-detected if None)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information

    Example:
        >>> # Sequence-level (with preprocessing)
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("agg", AggregateSequences("mean")),
        ...     ("probe", MLP(hidden_dim=128)),
        ... ])
        >>> pipeline.fit(acts, labels)

        >>> # Token-level (no aggregation)
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("probe", MLP(hidden_dim=128)),
        ...     ("agg_scores", AggregateTokenScores("mean")),
        ... ])
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
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize MLP probe.

        Args:
            hidden_dim: Number of hidden units
            dropout: Dropout rate (None for no dropout)
            activation: Activation function ("relu" or "gelu")
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            n_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            device: Device for computation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        super().__init__(
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        # Architecture parameters
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._d_model = None
        self._trained_on_tokens = False

        # Streaming state
        self._streaming_steps = 0

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        """Initialize the network and optimizer once we know the input dimension."""
        self._d_model = d_model
        self._network = _MLPNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

        # Always use float32 for MLP weights to avoid numerical issues
        # with float16/bfloat16 models that have extreme activation values
        self._network = self._network.to(torch.float32)

        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X: Activations, y: list | torch.Tensor) -> "MLP":
        """Fit the probe on activations.

        Adapts to input dimensionality:
        - If X has SEQ axis: Trains on individual tokens
        - Otherwise: Trains on sequence-level features

        Args:
            X: Activations to train on
            y: Labels [batch] or [batch, seq]

        Returns:
            self: Fitted probe instance

        Raises:
            ValueError: If X has unexpected axes (e.g., LAYER axis)
        """
        # Convert labels to tensor
        y_tensor = self._to_tensor(y)

        # Check for LAYER axis
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                "MLP probe expects single layer activations. "
                "Use SelectLayer transformer in pipeline before probe."
            )

        # Adapt based on SEQ axis
        if X.has_axis(Axis.SEQ):
            # TOKEN-LEVEL training
            features, tokens_per_sample = X.extract_tokens()  # [n_tokens, hidden]

            # Handle labels (ensure device compatibility)
            if y_tensor.ndim == 1:
                # Labels are [batch] → expand to [n_tokens]
                labels = torch.repeat_interleave(y_tensor, tokens_per_sample.cpu())
            elif y_tensor.ndim == 2:
                # Labels are [batch, seq] → extract tokens
                labels = y_tensor[X.detection_mask.cpu().bool()]
            else:
                raise ValueError(
                    f"Invalid label shape: {y_tensor.shape}. "
                    f"Expected [batch] or [batch, seq]"
                )

            self._trained_on_tokens = True
            self._tokens_per_sample = tokens_per_sample
        else:
            # SEQUENCE-LEVEL training
            features = X.activations  # [batch, hidden]
            labels = y_tensor  # [batch]

            if labels.ndim != 1:
                raise ValueError(
                    f"Expected 1D labels for sequence-level training, "
                    f"got shape {labels.shape}"
                )

            self._trained_on_tokens = False
            self._tokens_per_sample = None

        # Skip if no features
        if features.shape[0] == 0:
            if self.verbose:
                print("No features to train on (empty batch)")
            return self

        # Move to device and convert to float32 to avoid numerical issues
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Initialize network if needed
        if self._network is None:
            self._init_network(features.shape[1], dtype=torch.float32)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Training loop
        self._network.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            n_batches = 0

            for batch_features, batch_labels in dataloader:
                # Forward pass
                self._optimizer.zero_grad()
                outputs = self._network(batch_features)
                loss = F.binary_cross_entropy_with_logits(outputs, batch_labels)

                # Backward pass
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch + 1}/{self.n_epochs}: loss={avg_loss:.4f}")

        self._network.eval()
        self._fitted = True
        return self

    def partial_fit(self, X: Activations, y: list | torch.Tensor) -> "MLP":
        """Perform single gradient step for streaming/online learning.

        Args:
            X: Batch of activations
            y: Batch of labels

        Returns:
            self: Updated probe instance
        """
        # Convert labels to tensor
        y_tensor = self._to_tensor(y)

        # Check for LAYER axis
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                "MLP probe expects single layer activations. "
                "Use SelectLayer transformer in pipeline before probe."
            )

        # Extract features based on dimensionality
        if X.has_axis(Axis.SEQ):
            # Token-level training
            features, tokens_per_sample = X.extract_tokens()

            # Expand labels if needed (ensure device compatibility)
            if y_tensor.ndim == 1:
                labels = torch.repeat_interleave(y_tensor, tokens_per_sample.cpu())
            elif y_tensor.ndim == 2:
                labels = y_tensor[X.detection_mask.cpu().bool()]
            else:
                raise ValueError(f"Invalid label shape: {y_tensor.shape}")

            self._trained_on_tokens = True
            self._tokens_per_sample = tokens_per_sample
        else:
            # Sequence-level training
            features = X.activations
            labels = y_tensor
            self._trained_on_tokens = False
            self._tokens_per_sample = None

        # Skip empty batches
        if features.shape[0] == 0:
            return self

        # Move to device
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.float32)

        # Initialize network if needed
        if self._network is None:
            self._init_network(features.shape[1])

        # Single gradient step
        self._network.train()
        self._optimizer.zero_grad()
        logits = self._network(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self._optimizer.step()

        self._streaming_steps += 1
        self._fitted = True
        return self

    def predict_proba(self, X: Activations) -> Scores:
        """Predict class probabilities.

        Returns Scores object matching input dimensionality:
        - If X has SEQ axis: Returns token-level Scores [batch, seq, 2]
        - Otherwise: Returns sequence-level Scores [batch, 2]

        Args:
            X: Activations to predict on

        Returns:
            Scores object with predictions

        Raises:
            ValueError: If probe not fitted or X has unexpected axes
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        # Check for LAYER axis
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                "Expected single layer activations. "
                "Use SelectLayer transformer in pipeline."
            )

        # Extract features based on dimensionality
        if X.has_axis(Axis.SEQ):
            # Token-level prediction
            features, tokens_per_sample = X.extract_tokens()  # [n_tokens, hidden]
            is_token_level = True
        else:
            # Sequence-level prediction
            features = X.activations  # [batch, hidden]
            tokens_per_sample = None
            is_token_level = False

        # Move to device and convert to float32 for consistency
        features = features.to(self.device, dtype=torch.float32)

        # Get predictions
        self._network.eval()
        with torch.no_grad():
            logits = self._network(features)
            probs_positive = torch.sigmoid(logits)

            # Create 2-class probability matrix
            probs = torch.stack([1 - probs_positive, probs_positive], dim=-1)

        # Wrap in Scores object
        if is_token_level:
            return Scores.from_token_scores(
                probs, tokens_per_sample, batch_indices=X.batch_indices
            )
        else:
            return Scores.from_sequence_scores(probs, batch_indices=X.batch_indices)

    def save(self, path: Path | str) -> None:
        """Save the probe to disk.

        Args:
            path: File path to save the probe

        Raises:
            RuntimeError: If probe not fitted
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted probe")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        state = {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "d_model": self._d_model,
            "network_state": self._network.state_dict(),
            "trained_on_tokens": self._trained_on_tokens,
        }

        torch.save(state, path)

        if self.verbose:
            print(f"Probe saved to {path}")

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "MLP":
        """Load a probe from disk.

        Args:
            path: File path to load the probe from
            device: Device to load the probe onto (auto-detected if None)

        Returns:
            Loaded probe instance
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu")

        # Create probe instance
        probe = cls(
            hidden_dim=state["hidden_dim"],
            dropout=state.get("dropout"),
            activation=state["activation"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            batch_size=state["batch_size"],
            device=device or state.get("device"),
            random_state=state.get("random_state"),
            verbose=state.get("verbose", False),
        )

        # Initialize network and load state
        probe._d_model = state["d_model"]
        probe._init_network(probe._d_model)
        probe._network.load_state_dict(state["network_state"])
        probe._network.eval()

        # Restore training state
        probe._trained_on_tokens = state.get("trained_on_tokens", False)
        probe._fitted = True

        return probe

    def __repr__(self) -> str:
        """String representation of the probe."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        token_str = ", token-level" if self._trained_on_tokens else ""
        return f"MLP(hidden_dim={self.hidden_dim}, {fitted_str}{token_str})"
