"""
Attention-based probe implementation following sklearn-style API.

This probe learns attention weights over the sequence dimension to focus on
the most relevant parts for classification, instead of using fixed aggregation.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations, Axis
from ..processing.scores import Scores
from .base import BaseProbe


class _AttentionNetwork(nn.Module):
    """
    Attention-based neural network for sequence classification (internal implementation).

    Architecture:
    - Attention scoring: MLP that outputs attention scores for each token
    - Attention weights: Softmax over scores (masked by detection mask)
    - Weighted aggregation: Sum of attention-weighted activations
    - Classifier: MLP that predicts from the aggregated representation
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.temperature = temperature

        # Attention scoring module with layer norm and dropout
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Classifier with dropout for regularization
        self.classifier_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2),  # Larger hidden dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(
                    module.weight, gain=0.5
                )  # Smaller gain for better init
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention mechanism.

        Args:
            sequences: Input sequences [batch, seq_len, d_model]
            mask: Valid token mask [batch, seq_len]

        Returns:
            Tuple of (logits, attention_weights)
            - logits: Classification logits [batch]
            - attention_weights: Attention weights [batch, seq_len]
        """
        # Apply layer norm before attention scoring
        normed_sequences = self.attention_norm(sequences)

        # Compute attention scores for each token
        attention_scores = self.attention_scorer(normed_sequences).squeeze(
            -1
        )  # [batch, seq_len]

        # Apply temperature scaling for better calibration
        attention_scores = attention_scores / self.temperature

        # Apply mask to attention scores
        attention_scores_masked = attention_scores.masked_fill(
            ~mask.bool(), float("-inf")
        )

        # Compute attention weights via softmax
        attention_weights = torch.softmax(attention_scores_masked, dim=1)

        # Handle edge case where all positions are masked
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Apply attention weights to get weighted representation
        weighted = (
            attention_weights.unsqueeze(-1) * sequences
        )  # [batch, seq_len, d_model]
        aggregated = weighted.sum(dim=1)  # [batch, d_model]

        # Apply layer norm before classification
        aggregated = self.classifier_norm(aggregated)

        # Classify the aggregated representation
        logits = self.classifier(aggregated).squeeze(-1)  # [batch]

        return logits, attention_weights


class Attention(BaseProbe):
    """
    Attention-based probe for sequence classification.

    Instead of using simple aggregation (mean/max), this probe learns attention
    weights to focus on the most relevant parts of the sequence for classification.

    **IMPORTANT**: This probe REQUIRES sequences (SEQ axis must be present).
    Use SelectLayer (not AggregateSequences) before this probe in the pipeline.

    The probe returns [batch, 2] scores by applying learned attention over sequences.

    Args:
        hidden_dim: Hidden dimension for attention and classifier networks
        dropout: Dropout rate for regularization
        temperature: Temperature scaling for attention softmax
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: L2 regularization strength
        n_epochs: Maximum number of training epochs
        patience: Early stopping patience
        device: Device for computation (auto-detected if None)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information

    Example:
        >>> # Attention requires sequences - do NOT use AggregateSequences
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),  # Keep SEQ axis
        ...     ("probe", Attention(hidden_dim=128)),  # Learns attention weights
        ... ])
        >>> pipeline.fit(acts, labels)
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
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize attention probe.

        Args:
            hidden_dim: Hidden dimension for networks
            dropout: Dropout rate for regularization
            temperature: Temperature scaling for attention softmax
            learning_rate: Learning rate for AdamW
            weight_decay: Weight decay for AdamW (L2 regularization)
            n_epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device for PyTorch operations
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        super().__init__(
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._d_model = None
        self.attention_weights = None  # Store for interpretability

        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        """Initialize the network once we know the input dimension."""
        self._d_model = d_model
        self._network = _AttentionNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            temperature=self.temperature,
        ).to(self.device)

        # Match the dtype of the input features for mixed precision support
        if dtype is not None:
            self._network = self._network.to(dtype)

        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.startswith("cuda"),
        )

    def fit(self, X: Activations, y: list | torch.Tensor) -> "Attention":
        """
        Fit the probe on training data.

        The Attention probe REQUIRES sequences (SEQ axis must be present).
        It trains by learning attention weights over the sequence dimension.

        Args:
            X: Activations to train on (must have SEQ axis)
            y: Labels [batch]

        Returns:
            self: Fitted probe instance

        Raises:
            ValueError: If X doesn't have SEQ axis or has LAYER axis
        """
        # Convert labels to tensor
        y_tensor = self._to_tensor(y)

        # Check for LAYER axis
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                "Attention probe expects single layer activations. "
                "Use SelectLayer transformer in pipeline before probe."
            )

        # Check for SEQ axis (REQUIRED)
        if not X.has_axis(Axis.SEQ):
            raise ValueError(
                "Attention probe requires sequences (SEQ axis). "
                "Do not use AggregateSequences before this probe. "
                "The probe learns attention weights over sequences internally."
            )

        # Extract sequences and detection mask
        sequences = X.activations  # [batch, seq, hidden]
        detection_mask = X.detection_mask  # [batch, seq]
        labels = y_tensor  # [batch]

        if labels.ndim != 1:
            raise ValueError(
                f"Expected 1D labels for attention probe, got shape {labels.shape}"
            )

        # Move to device and ensure tensors are safe for autograd
        sequences = sequences.to(self.device)
        # Clone to avoid issues with inference tensors in autograd
        sequences = sequences.clone()
        detection_mask = detection_mask.to(self.device)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Validate we have both classes
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Training data must contain both classes. Found: {unique_labels.tolist()}"
            )

        # Initialize network if needed
        if self._network is None:
            self._init_network(sequences.shape[-1], dtype=sequences.dtype)

        # Create train/validation split
        n_samples = len(sequences)
        n_val = max(1, int(0.2 * n_samples))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        indices = torch.randperm(n_samples, device=self.device)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        train_sequences = sequences[train_indices]
        train_mask = detection_mask[train_indices]
        train_y = labels[train_indices]

        val_sequences = sequences[val_indices]
        val_mask = detection_mask[val_indices]
        val_y = labels[val_indices]

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        self._network.train()
        for epoch in range(self.n_epochs):
            # Training step
            self._optimizer.zero_grad()
            logits, _ = self._network(train_sequences, train_mask)
            loss = F.binary_cross_entropy_with_logits(logits, train_y)
            loss.backward()
            self._optimizer.step()

            # Validation step (less frequent for efficiency)
            if epoch % 10 == 0:
                self._network.eval()
                with torch.no_grad():
                    val_logits, _ = self._network(val_sequences, val_mask)
                    val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y)
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
                if val_loss < 0.001:  # Reduced from 0.01 for better convergence
                    if self.verbose:
                        print(f"Converged at epoch {epoch}")
                    break

        self._network.eval()
        self._fitted = True
        return self

    def predict_proba(self, X: Activations) -> Scores:
        """
        Predict class probabilities using attention mechanism.

        Returns sequence-level Scores [batch, 2] by applying learned attention
        over the sequence dimension.

        Args:
            X: Activations to predict on (must have SEQ axis)

        Returns:
            Scores object with predictions [batch, 2]

        Raises:
            ValueError: If probe not fitted or X doesn't have SEQ axis
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        # Check for LAYER axis
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                "Expected single layer activations. "
                "Use SelectLayer transformer in pipeline."
            )

        # Check for SEQ axis (REQUIRED)
        if not X.has_axis(Axis.SEQ):
            raise ValueError(
                "Attention probe requires sequences (SEQ axis). "
                "Do not use Pool(dim='sequence') before this probe."
            )

        # Extract sequences and detection mask
        sequences = X.activations  # [batch, seq, hidden]
        detection_mask = X.detection_mask  # [batch, seq]

        # Move to device
        sequences = sequences.to(self.device)
        detection_mask = detection_mask.to(self.device)

        # Get predictions
        self._network.eval()
        with torch.no_grad():
            logits, attention_weights = self._network(sequences, detection_mask)

            # Store attention weights for interpretability
            self.attention_weights = attention_weights.detach().cpu()

            # Convert logits to probabilities
            probs_positive = torch.sigmoid(logits)

            # Create 2-class probability matrix
            probs = torch.stack([1 - probs_positive, probs_positive], dim=-1)

        # Return sequence-level scores (attention does its own aggregation)
        return Scores.from_sequence_scores(probs, batch_indices=X.batch_indices)

    def save(self, path: Path | str) -> None:
        """
        Save the probe to disk.

        Args:
            path: Path to save the probe

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
            "temperature": self.temperature,
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
        }

        torch.save(state, path)

        if self.verbose:
            print(f"Probe saved to {path}")

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "Attention":
        """
        Load a probe from disk.

        Args:
            path: Path to load the probe from
            device: Device to load onto (auto-detected if None)

        Returns:
            Loaded probe instance
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu")

        # Create probe instance
        probe = cls(
            hidden_dim=state["hidden_dim"],
            dropout=state.get("dropout", 0.2),
            temperature=state.get("temperature", 2.0),
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            patience=state["patience"],
            device=device or state.get("device"),
            random_state=state.get("random_state"),
            verbose=state.get("verbose", False),
        )

        # Initialize network
        probe._d_model = state["d_model"]
        probe._init_network(probe._d_model)

        # Load network and optimizer states
        probe._network.load_state_dict(state["network_state_dict"])
        probe._optimizer.load_state_dict(state["optimizer_state_dict"])
        probe._network.eval()

        # Move to correct device
        probe._network = probe._network.to(probe.device)
        probe._fitted = True

        return probe

    def __repr__(self) -> str:
        """String representation of the probe."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        return (
            f"Attention(hidden_dim={self.hidden_dim}, "
            f"attention-based, {fitted_str})"
        )
