"""
GPU-based logistic regression probe implementation.
Follows the same pattern as MLP probe for consistency and performance.
"""

from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from ..processing.activations import ActivationIterator, Activations
from .base import BaseProbe


class LogisticNetwork(nn.Module):
    """
    Simple linear model for GPU-based logistic regression.

    This is equivalent to sklearn's LogisticRegression with fit_intercept=False.
    Uses a single linear layer without bias for binary classification.
    """

    def __init__(self, d_model: int):
        """
        Initialize the logistic regression network.

        Args:
            d_model: Input dimension (hidden size of the model)
        """
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=False)

        # Initialize weights with small random values (similar to sklearn)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer.

        Args:
            x: Input tensor [batch_size, d_model]

        Returns:
            Logits tensor [batch_size]
        """
        return self.linear(x).squeeze(-1)


class GPUStandardScaler:
    """
    GPU-based standardization with support for both batch and streaming modes.

    Maintains running statistics for streaming updates using Welford's algorithm.
    All operations are performed on the specified device for efficiency.
    """

    def __init__(self, device: str = "cuda", epsilon: float = 1e-8):
        """
        Initialize the GPU standard scaler.

        Args:
            device: Device to use for computations
            epsilon: Small value for numerical stability
        """
        self.device = device
        self.epsilon = epsilon

        # Statistics (initialized on first fit)
        self.mean_: torch.Tensor | None = None
        self.var_: torch.Tensor | None = None
        self.scale_: torch.Tensor | None = None
        self.n_samples_seen_: int = 0

        # For streaming updates (Welford's algorithm)
        self.mean_accumulator_: torch.Tensor | None = None
        self.m2_accumulator_: torch.Tensor | None = None

    def fit(self, X: torch.Tensor) -> "GPUStandardScaler":
        """
        Compute mean and standard deviation for standardization.

        Args:
            X: Training data [n_samples, n_features]

        Returns:
            self: Fitted scaler
        """
        X = X.to(self.device)

        # Compute statistics in one pass
        self.mean_ = X.mean(dim=0)
        self.var_ = X.var(dim=0, unbiased=False)  # Use biased variance like sklearn
        self.scale_ = torch.sqrt(self.var_ + self.epsilon)
        self.n_samples_seen_ = X.shape[0]

        # Initialize streaming accumulators
        self.mean_accumulator_ = self.mean_.clone()
        self.m2_accumulator_ = self.var_ * self.n_samples_seen_

        return self

    def partial_fit(self, X: torch.Tensor) -> "GPUStandardScaler":
        """
        Incrementally update statistics using vectorized Welford's algorithm.

        Args:
            X: Batch of training data [n_samples, n_features]

        Returns:
            self: Updated scaler
        """
        X = X.to(self.device)
        n_samples = X.shape[0]

        if self.mean_ is None:
            # First batch - initialize
            return self.fit(X)

        # Vectorized Welford's algorithm - much faster than loop
        batch_mean = X.mean(dim=0)
        batch_var = X.var(dim=0, unbiased=False)

        # Combine statistics using parallel algorithm
        delta = batch_mean - self.mean_accumulator_
        total_samples = self.n_samples_seen_ + n_samples

        # Update mean
        self.mean_accumulator_ = (
            self.mean_accumulator_ * self.n_samples_seen_ + batch_mean * n_samples
        ) / total_samples

        # Update M2 (sum of squared deviations)
        self.m2_accumulator_ = (
            self.m2_accumulator_
            + batch_var * n_samples
            + delta**2 * self.n_samples_seen_ * n_samples / total_samples
        )

        self.n_samples_seen_ = total_samples

        # Update statistics
        self.mean_ = self.mean_accumulator_.clone()
        if self.n_samples_seen_ > 1:
            self.var_ = self.m2_accumulator_ / self.n_samples_seen_
            self.scale_ = torch.sqrt(self.var_ + self.epsilon)

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Standardize features by removing mean and scaling to unit variance.

        Args:
            X: Data to transform [n_samples, n_features]

        Returns:
            Standardized data
        """
        if self.mean_ is None:
            raise RuntimeError("Scaler must be fitted before transform")

        X = X.to(self.device)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit to data, then transform it.

        Args:
            X: Training data [n_samples, n_features]

        Returns:
            Standardized data
        """
        return self.fit(X).transform(X)


class Logistic(BaseProbe):
    """
    GPU-based logistic regression probe for binary classification.

    This probe uses PyTorch's autograd for training, making it efficient
    on GPU especially for token-level training (score_aggregation mode).
    It follows the same pattern as MLP probe for consistency.

    Attributes:
        l2_penalty: L2 regularization strength (weight_decay in optimizer)
        learning_rate: Learning rate for optimizer
        n_epochs: Maximum number of training epochs
        patience: Early stopping patience
        _network: LogisticNetwork instance
        _optimizer: PyTorch optimizer
        _scaler: GPUStandardScaler instance
    """

    def __init__(
        self,
        layer: int,
        sequence_aggregation: Literal["mean", "max", "last_token"] | None = None,
        score_aggregation: Literal["mean", "max", "last_token"] | None = None,
        l2_penalty: float = 1.0,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize GPU-based logistic regression probe.

        Args:
            layer: Layer index to use activations from
            sequence_aggregation: Aggregate sequences BEFORE training
            score_aggregation: Train on tokens, aggregate AFTER prediction
            l2_penalty: L2 regularization strength (1/C in sklearn)
            learning_rate: Learning rate (None for automatic)
            n_epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device for PyTorch operations
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        # Validate that at least one aggregation is specified
        if sequence_aggregation is None and score_aggregation is None:
            raise ValueError(
                "Logistic requires either sequence_aggregation or score_aggregation"
            )

        super().__init__(
            layer=layer,
            sequence_aggregation=sequence_aggregation,
            score_aggregation=score_aggregation,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        self.l2_penalty = l2_penalty
        self.n_epochs = n_epochs
        self.patience = patience
        self.learning_rate = learning_rate

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._scaler = GPUStandardScaler(device=self.device)
        self._d_model = None
        self._streaming_steps = 0

        # This probe requires gradients for training
        self._requires_grad = True

        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

    def _init_network(self, d_model: int):
        """Initialize the network and optimizer once we know the input dimension."""
        self._d_model = d_model
        self._network = LogisticNetwork(d_model).to(self.device)

        # Convert l2_penalty to weight_decay (they're equivalent)
        weight_decay = self.l2_penalty

        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
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

        # Skip if no features
        if features.shape[0] == 0:
            if self.verbose:
                print("No features to train on (empty batch)")
            return self

        # Move to device
        features = features.to(self.device).float()
        labels = labels.to(self.device).float()

        # Standardize features
        features_scaled = self._scaler.fit_transform(features)

        # Initialize network if needed
        if self._network is None:
            self._init_network(features.shape[1])

        # Create train/validation split for early stopping
        n_samples = len(features_scaled)
        n_val = max(1, int(0.2 * n_samples))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        indices = torch.randperm(n_samples, device=self.device)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        X_train = features_scaled[train_indices]
        y_train = labels[train_indices]
        X_val = features_scaled[val_indices]
        y_val = labels[val_indices]

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

            # Validation and early stopping
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

        # Skip empty batches (no detected tokens)
        if features.shape[0] == 0:
            if self.verbose:
                print("Skipping batch with no detected tokens")
            return self

        # Move to device
        features = features.to(self.device).float()
        labels = labels.to(self.device).float()

        # Update scaler and transform
        if self._scaler.mean_ is None:
            features_scaled = self._scaler.fit_transform(features)
        else:
            self._scaler.partial_fit(features)
            features_scaled = self._scaler.transform(features)

        # Initialize network on first batch
        if self._network is None:
            self._init_network(features.shape[1])

        # Training step
        self._network.train()
        self._optimizer.zero_grad()
        logits = self._network(features_scaled)
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

        # Get features based on training mode
        features = self._prepare_features(X)
        features = features.to(self.device).float()

        # Standardize features
        features_scaled = self._scaler.transform(features)

        # Get predictions
        self._network.eval()
        with torch.no_grad():
            logits = self._network(features_scaled)

            # Apply score aggregation if needed
            if self.score_aggregation is not None and hasattr(
                self, "_tokens_per_sample"
            ):
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
            "l2_penalty": self.l2_penalty,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "d_model": self._d_model,
            "network_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scaler_mean": self._scaler.mean_,
            "scaler_var": self._scaler.var_,
            "scaler_scale": self._scaler.scale_,
            "scaler_n_samples": self._scaler.n_samples_seen_,
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
            l2_penalty=state["l2_penalty"],
            learning_rate=state["learning_rate"],
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

        # Load scaler state
        probe._scaler.mean_ = state["scaler_mean"].to(probe.device)
        probe._scaler.var_ = state["scaler_var"].to(probe.device)
        probe._scaler.scale_ = state["scaler_scale"].to(probe.device)
        probe._scaler.n_samples_seen_ = state["scaler_n_samples"]
        probe._scaler.mean_accumulator_ = probe._scaler.mean_.clone()
        probe._scaler.m2_accumulator_ = (
            probe._scaler.var_ * probe._scaler.n_samples_seen_
        )

        probe._streaming_steps = state["streaming_steps"]

        # Move to correct device
        probe._network = probe._network.to(probe.device)
        probe._network.eval()
        probe._fitted = True

        return probe


class SklearnLogistic(BaseProbe):
    """
    Logistic regression probe with L2 regularization.

    This probe uses sklearn's LogisticRegression for batch training
    and SGDClassifier for streaming/incremental training. Features are
    automatically standardized before training.

    Attributes:
        l2_penalty: L2 regularization strength (default 1.0)
        _clf: Underlying sklearn classifier
        _scaler: StandardScaler for feature normalization
        _coef: Coefficient vector as tensor for efficient prediction
        _scaler_mean: Scaler mean as tensor
        _scaler_scale: Scaler scale as tensor
        _use_sgd: Whether using SGD (streaming) or regular LogisticRegression
    """

    def __init__(
        self,
        layer: int,
        sequence_aggregation: str | None = None,
        score_aggregation: str | None = None,
        l2_penalty: float = 1.0,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize logistic regression probe.

        Args:
            layer: Layer index to use activations from
            sequence_aggregation: Aggregate sequences BEFORE training (classic)
            score_aggregation: Train on tokens, aggregate AFTER prediction
            l2_penalty: L2 regularization strength (C = 1/l2_penalty in sklearn)
            device: Device for PyTorch operations
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        # Validate that at least one aggregation is specified
        if sequence_aggregation is None and score_aggregation is None:
            raise ValueError(
                "LogisticProbe requires either sequence_aggregation or score_aggregation"
            )

        super().__init__(
            layer=layer,
            sequence_aggregation=sequence_aggregation,
            score_aggregation=score_aggregation,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        self.l2_penalty = l2_penalty
        self._clf = None
        self._scaler = StandardScaler()
        self._coef = None
        self._scaler_mean = None
        self._scaler_scale = None
        self._use_sgd = False

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

        features = self._prepare_features(X)
        labels = self._prepare_labels(
            y, expand_for_tokens=(self.sequence_aggregation is None)
        )

        # Convert to numpy
        X_np = features.cpu().float().numpy()
        y_np = labels.cpu().numpy()

        # Validate we have both classes
        unique_classes = np.unique(y_np)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Training data must contain both classes. Found: {unique_classes}"
            )

        # Standardize features
        X_scaled = self._scaler.fit_transform(X_np)

        # Train logistic regression
        self._clf = LogisticRegression(
            C=1.0 / self.l2_penalty,  # sklearn uses C = 1/lambda
            fit_intercept=False,
            random_state=self.random_state,
        )
        self._clf.fit(X_scaled, y_np)

        # Cache coefficients as tensors for fast prediction
        self._coef = torch.tensor(
            self._clf.coef_[0], device=self.device, dtype=torch.float32
        )
        self._scaler_mean = torch.tensor(
            self._scaler.mean_, device=self.device, dtype=torch.float32
        )
        self._scaler_scale = torch.tensor(
            self._scaler.scale_, device=self.device, dtype=torch.float32
        )

        self._fitted = True
        self._use_sgd = False
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

        Uses SGDClassifier for online learning.

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

        # Convert to numpy
        X_np = features.cpu().float().numpy()
        y_np = labels.cpu().numpy()

        if self._clf is None:
            # First batch - initialize SGD classifier
            self._clf = SGDClassifier(
                loss="log_loss",  # Logistic loss
                # TODO: try empirically what works here
                alpha=self.l2_penalty,  # Scale by batch size
                fit_intercept=False,
                random_state=42,
            )

            # Fit scaler on first batch
            X_scaled = self._scaler.fit_transform(X_np)
            self._scaler_mean_f32 = self._scaler.mean_.astype(np.float32)
            self._scaler_scale_f32 = self._scaler.scale_.astype(np.float32)

            # Initial fit with explicit classes
            self._clf.partial_fit(X_scaled, y_np, classes=[0, 1])

            self._use_sgd = True
        else:
            # Subsequent batches
            if not self._use_sgd:
                raise ValueError(
                    "Do not combine fit() and partial_fit() for Logistic probe"
                )
            X_scaled = (X_np - self._scaler_mean_f32) / self._scaler_scale_f32
            self._clf.partial_fit(X_scaled, y_np)

        # Update cached tensors
        self._coef = torch.tensor(
            self._clf.coef_[0], device=self.device, dtype=torch.float32
        )

        # Also set tensor versions for predict_proba
        if self._scaler_mean is None:
            self._scaler_mean = torch.tensor(
                self._scaler_mean_f32, device=self.device, dtype=torch.float32
            )
            self._scaler_scale = torch.tensor(
                self._scaler_scale_f32, device=self.device, dtype=torch.float32
            )

        # Mark as fitted since we've successfully trained on at least one batch
        self._fitted = True
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

        # Get features based on training mode
        features = self._prepare_features(X)
        original_device = features.device

        # Only move parameters if they're not already on the correct device
        if self._scaler_mean.device != original_device:
            scaler_mean = self._scaler_mean.to(original_device)
            scaler_scale = self._scaler_scale.to(original_device)
            coef = self._coef.to(original_device)
        else:
            scaler_mean = self._scaler_mean
            scaler_scale = self._scaler_scale
            coef = self._coef

        # Standardize (on the same device as features)
        features_scaled = (features - scaler_mean) / scaler_scale

        # Compute logits
        logits = features_scaled @ coef

        # Apply score aggregation if needed (aggregate logits, not probabilities)
        if self.score_aggregation is not None and hasattr(self, "_tokens_per_sample"):
            # Aggregate logits before applying sigmoid
            logits = self._aggregate_scores(
                logits.unsqueeze(-1), self.score_aggregation
            ).squeeze(-1)

        # Get probabilities
        probs_positive = torch.sigmoid(logits)

        # Create 2-class probability matrix (on same device as input)
        n_predictions = len(probs_positive)
        probs = torch.zeros(n_predictions, 2, device=original_device)
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
            "l2_penalty": self.l2_penalty,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "coef": self._coef.cpu(),
            "scaler_mean": self._scaler.mean_,
            "scaler_scale": self._scaler.scale_,
            "scaler_n_samples_seen": self._scaler.n_samples_seen_,
            "use_sgd": self._use_sgd,
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
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Create probe instance
        probe = cls(
            layer=state["layer"],
            sequence_aggregation=state["sequence_aggregation"],
            score_aggregation=state["score_aggregation"],
            l2_penalty=state["l2_penalty"],
            device=device or state["device"],
            random_state=state["random_state"],
            verbose=state.get("verbose", False),
        )

        # Restore model state
        probe._coef = state["coef"].to(probe.device)
        probe._scaler_mean = torch.tensor(
            state["scaler_mean"], device=probe.device, dtype=torch.float32
        )
        probe._scaler_scale = torch.tensor(
            state["scaler_scale"], device=probe.device, dtype=torch.float32
        )

        # Restore scaler
        probe._scaler.mean_ = state["scaler_mean"]
        probe._scaler.scale_ = state["scaler_scale"]
        probe._scaler.n_samples_seen_ = state["scaler_n_samples_seen"]

        # Create appropriate classifier
        probe._use_sgd = state["use_sgd"]
        if probe._use_sgd:
            probe._clf = SGDClassifier(
                loss="log_loss",
                alpha=probe.l2_penalty / 1000,
                fit_intercept=False,
                random_state=probe.random_state,
            )
            probe._clf.coef_ = probe._coef.cpu().numpy().reshape(1, -1)
            probe._clf.intercept_ = np.zeros(1)
            probe._clf.classes_ = np.array([0, 1])
        else:
            probe._clf = LogisticRegression(
                C=1.0 / probe.l2_penalty,
                fit_intercept=False,
                random_state=probe.random_state,
            )
            probe._clf.coef_ = probe._coef.cpu().numpy().reshape(1, -1)
            probe._clf.intercept_ = np.zeros(1)
            probe._clf.classes_ = np.array([0, 1])

        probe._fitted = True
        return probe
