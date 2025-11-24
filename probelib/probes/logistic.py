"""GPU-accelerated L2-regularized logistic regression probe."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..processing.activations import Activations, Axis
from .base import BaseProbe


class _LogisticNetwork(nn.Module):
    """Simple logistic regression network (internal implementation)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class _GPUStandardScaler:
    """Standard scaler for GPU tensors."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X: torch.Tensor) -> "_GPUStandardScaler":
        """Fit the scaler on data."""
        X = X.to(self.device)
        self.mean_ = X.mean(dim=0)
        self.std_ = X.std(dim=0).clamp(min=1e-8)
        self._fitted = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform the data."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        X = X.to(self.device)
        mean = self.mean_.to(X.dtype)
        std = self.std_.to(X.dtype)
        return (X - mean) / std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class Logistic(BaseProbe):
    """GPU-accelerated L2-regularized logistic regression probe.

    This probe adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores

    Uses LBFGS optimizer for batch training.

    Args:
        C: Inverse of regularization strength (higher = less regularization)
        max_iter: Maximum number of iterations for LBFGS
        device: Device for computation (auto-detected if None)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information

    Example:
        >>> # Sequence-level (with preprocessing)
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("agg", AggregateSequences("mean")),
        ...     ("probe", Logistic(C=1.0)),
        ... ])
        >>> pipeline.fit(acts, labels)

        >>> # Token-level (no aggregation)
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("probe", Logistic(C=1.0)),
        ...     ("agg_scores", AggregateTokenScores("mean")),
        ... ])
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 100,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize logistic regression probe.

        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations for LBFGS
            device: Device for PyTorch operations
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        super().__init__(
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        # Training parameters
        self.C = C
        self.max_iter = max_iter

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._scaler = _GPUStandardScaler(device=self.device)
        self._d_model = None
        self._trained_on_tokens = False

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)

    def fit(self, X: Activations, y: list | torch.Tensor) -> "Logistic":
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
                "Logistic probe expects single layer activations. "
                "Use SelectLayer transformer in pipeline before probe."
            )

        # Adapt based on SEQ axis
        if X.has_axis(Axis.SEQ):
            # TOKEN-LEVEL training
            features, tokens_per_sample = X.extract_tokens()  # [n_tokens, hidden]

            # Handle labels
            if y_tensor.ndim == 1:
                # Labels are [batch] → expand to [n_tokens]
                labels = torch.repeat_interleave(y_tensor, tokens_per_sample)
            elif y_tensor.ndim == 2:
                # Labels are [batch, seq] → extract tokens
                labels = y_tensor[X.detection_mask.bool()]
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

        # Move to device
        features = features.to(self.device)
        labels = labels.to(self.device).float()

        # Initialize network if needed
        if self._network is None:
            d_model = features.shape[1]
            self._d_model = d_model
            self._network = _LogisticNetwork(d_model).to(self.device)

            if features.dtype != torch.float32:
                self._network = self._network.to(features.dtype)

            self._optimizer = torch.optim.LBFGS(
                self._network.parameters(),
                max_iter=self.max_iter,
                line_search_fn="strong_wolfe",
            )

        # Standardize features
        features_scaled = self._scaler.fit_transform(features)

        # Compute L2 regularization weight
        n_samples = features.shape[0]
        l2_weight = 1.0 / (2.0 * self.C * n_samples) if self.C > 0 else 0.0

        self._network.train()

        # LBFGS requires a closure
        def closure():
            self._optimizer.zero_grad()
            logits = self._network(features_scaled)

            # Binary cross entropy loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Add L2 regularization on weights (not bias)
            l2_reg = 0.0
            if l2_weight > 0:
                for name, param in self._network.named_parameters():
                    if "weight" in name:
                        l2_reg += torch.sum(param**2)

            loss = bce_loss + l2_weight * l2_reg
            loss.backward()
            return loss

        # Run LBFGS optimization
        self._optimizer.step(closure)

        if self.verbose:
            # Evaluate final loss
            self._network.eval()
            with torch.no_grad():
                logits = self._network(features_scaled)
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
                print(f"LBFGS converged with BCE loss: {bce_loss.item():.4f}")

        self._network.eval()
        self._fitted = True
        return self

    def predict_proba(self, X: Activations) -> torch.Tensor:
        """Predict class probabilities.

        Returns scores matching input dimensionality:
        - If X has SEQ axis: Returns [n_tokens, 2]
        - Otherwise: Returns [batch, 2]

        Args:
            X: Activations to predict on

        Returns:
            Predicted probabilities

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
            features, _ = X.extract_tokens()  # [n_tokens, hidden]
        else:
            # Sequence-level prediction
            features = X.activations  # [batch, hidden]

        # Move to device and standardize
        features = features.to(self.device)
        features_scaled = self._scaler.transform(features)

        # Get predictions
        self._network.eval()
        with torch.no_grad():
            logits = self._network(features_scaled)
            probs_positive = torch.sigmoid(logits)

            # Create 2-class probability matrix
            probs = torch.stack([1 - probs_positive, probs_positive], dim=-1)

        return probs  # [n_tokens, 2] or [batch, 2]

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
            "C": self.C,
            "max_iter": self.max_iter,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "network_state": self._network.state_dict(),
            "scaler_mean": self._scaler.mean_,
            "scaler_std": self._scaler.std_,
            "d_model": self._d_model,
            "trained_on_tokens": self._trained_on_tokens,
        }

        torch.save(state, path)

        if self.verbose:
            print(f"Probe saved to {path}")

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "Logistic":
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
            C=state["C"],
            max_iter=state["max_iter"],
            device=device or state.get("device"),
            random_state=state.get("random_state"),
            verbose=state.get("verbose", False),
        )

        # Restore network
        probe._d_model = state["d_model"]
        probe._network = _LogisticNetwork(probe._d_model).to(probe.device)
        probe._network.load_state_dict(state["network_state"])
        probe._network.eval()

        # Restore scaler
        probe._scaler.mean_ = state["scaler_mean"].to(probe.device)
        probe._scaler.std_ = state["scaler_std"].to(probe.device)
        probe._scaler._fitted = True

        # Restore training state
        probe._trained_on_tokens = state.get("trained_on_tokens", False)
        probe._fitted = True

        return probe

    def __repr__(self) -> str:
        """String representation of the probe."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        token_str = ", token-level" if self._trained_on_tokens else ""
        return f"Logistic(C={self.C}, {fitted_str}{token_str})"
