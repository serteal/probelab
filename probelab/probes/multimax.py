"""
Multi-head hard max pooling probe from GDM paper.

This probe uses H independent heads, each selecting the highest-scoring token
via hard max (not softmax). Helps with long-context generalization
by preventing signal dilution.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import Activations, Axis
from ..processing.scores import Scores
from .base import BaseProbe


class _MultiMaxNetwork(nn.Module):
    """Multi-head hard max pooling network (internal implementation).

    From GDM paper (Section 3.2.1):
        f_MultiMax(S) = sum_h max_j [v_h^T * MLP(x_j)]

    Architecture:
    - Shared MLP: Projects each token to a hidden representation
    - Per-head value projections: Each head computes a scalar score per token
    - Hard max pooling: Each head selects the single highest-scoring token
    - Output layer: Combines head outputs for final prediction
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 10,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_heads = n_heads

        # Shared MLP for all heads
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        # Per-head value projections (each outputs a scalar per token)
        self.head_projections = nn.ModuleList(
            [nn.Linear(mlp_hidden_dim, 1) for _ in range(n_heads)]
        )

        # Output layer to combine head outputs
        self.output = nn.Linear(n_heads, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with multi-head hard max pooling.

        Args:
            sequences: Input sequences [batch, seq_len, d_model]
            mask: Valid token mask [batch, seq_len]

        Returns:
            Logits [batch]
        """
        batch_size, seq_len, _ = sequences.shape

        # Apply shared MLP to all tokens
        normed = self.norm(sequences)
        mlp_out = self.mlp(normed)  # [batch, seq_len, mlp_hidden]

        # Compute per-head scores and apply hard max
        head_outputs = []
        for head_proj in self.head_projections:
            # Compute scores for this head
            scores = head_proj(mlp_out).squeeze(-1)  # [batch, seq_len]

            # Mask invalid positions
            scores_masked = scores.masked_fill(~mask.bool(), float("-inf"))

            # Hard max: select the single highest score per sample
            max_scores = scores_masked.max(dim=1).values  # [batch]
            head_outputs.append(max_scores)

        # Stack head outputs: [batch, n_heads]
        head_stack = torch.stack(head_outputs, dim=1)

        # Handle case where all positions are masked (all -inf -> set to 0)
        head_stack = torch.nan_to_num(head_stack, nan=0.0, posinf=0.0, neginf=0.0)

        # Combine heads with output layer
        logits = self.output(head_stack).squeeze(-1)  # [batch]

        return logits


class MultiMax(BaseProbe):
    """Multi-head hard max pooling probe from GDM paper.

    Uses H independent heads, each selecting the highest-scoring token
    via hard max (not softmax). Helps with long-context generalization
    by preventing signal dilution.

    From GDM paper (Section 3.2.1):
        f_MultiMax(S) = sum_h max_j [v_h^T * MLP(x_j)]

    **IMPORTANT**: This probe REQUIRES sequences (SEQ axis must be present).
    Use SelectLayer (not Pool) before this probe in the pipeline.

    Args:
        n_heads: Number of independent heads (default: 10)
        mlp_hidden_dim: Shared MLP hidden size (default: 128)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Learning rate (default: 5e-4)
        weight_decay: L2 regularization (default: 1e-3)
        n_epochs: Training epochs (default: 20)
        patience: Early stopping patience (default: 5)
        device: Device for computation (auto-detected if None)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information

    Example:
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("probe", MultiMax(n_heads=10)),
        ... ])
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
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize MultiMax probe."""
        super().__init__(
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        self.n_heads = n_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._d_model = None

        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        """Initialize the network once we know the input dimension."""
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
        """Fit the probe on training data.

        The MultiMax probe REQUIRES sequences (SEQ axis must be present).
        It trains by applying multi-head hard max pooling over the sequence.

        Args:
            X: Activations to train on (must have SEQ axis)
            y: Labels [batch]

        Returns:
            self: Fitted probe instance

        Raises:
            ValueError: If X doesn't have SEQ axis or has LAYER axis
        """
        y_tensor = self._to_tensor(y)

        # Check for LAYER axis
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                "MultiMax probe expects single layer activations. "
                "Use SelectLayer transformer in pipeline before probe."
            )

        # Check for SEQ axis (REQUIRED)
        if not X.has_axis(Axis.SEQ):
            raise ValueError(
                "MultiMax probe requires sequences (SEQ axis). "
                "Do not use Pool(dim='sequence') before this probe. "
                "The probe handles sequence aggregation via multi-head hard max."
            )

        # Extract sequences and detection mask
        sequences = X.activations  # [batch, seq, hidden]
        detection_mask = X.detection_mask  # [batch, seq]
        labels = y_tensor  # [batch]

        if labels.ndim != 1:
            raise ValueError(
                f"Expected 1D labels for MultiMax probe, got shape {labels.shape}"
            )

        # Keep data on CPU, move batches to device during training
        sequences = sequences.detach()
        detection_mask = detection_mask.detach()
        labels = labels.float()

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
        indices = torch.randperm(n_samples)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        # Mini-batch size for memory efficiency
        batch_size = min(32, len(train_indices))

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        self._network.train()
        for epoch in range(self.n_epochs):
            # Shuffle training data each epoch
            perm = torch.randperm(len(train_indices))
            shuffled_train_indices = train_indices[perm]

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(shuffled_train_indices), batch_size):
                batch_idx = shuffled_train_indices[i : i + batch_size]

                # Move batch to device
                batch_seq = sequences[batch_idx].to(self.device)
                batch_mask = detection_mask[batch_idx].to(self.device)
                batch_y = labels[batch_idx].to(self.device)

                self._optimizer.zero_grad()
                logits = self._network(batch_seq, batch_mask)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y)
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                # Free GPU memory
                del batch_seq, batch_mask, batch_y, logits, loss

            # Validation step
            self._network.eval()
            with torch.no_grad():
                val_seq = sequences[val_indices].to(self.device)
                val_mask = detection_mask[val_indices].to(self.device)
                val_y = labels[val_indices].to(self.device)
                val_logits = self._network(val_seq, val_mask)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y)
                del val_seq, val_mask, val_y, val_logits
            self._network.train()

            if self.verbose:
                print(f"  Epoch {epoch}: train_loss={epoch_loss/max(n_batches,1):.4f}, val_loss={val_loss.item():.4f}", flush=True)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}", flush=True)
                    break

            # Stop if loss is very small
            if val_loss < 0.001:
                if self.verbose:
                    print(f"Converged at epoch {epoch}", flush=True)
                break

        self._network.eval()
        self._fitted = True
        return self

    def predict_proba(self, X: Activations) -> Scores:
        """Predict class probabilities using multi-head hard max pooling.

        Returns sequence-level Scores [batch, 2] by applying multi-head
        hard max over the sequence dimension.

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
                "MultiMax probe requires sequences (SEQ axis). "
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
            logits = self._network(sequences, detection_mask)

            # Convert logits to probabilities
            probs_positive = torch.sigmoid(logits)

            # Create 2-class probability matrix
            probs = torch.stack([1 - probs_positive, probs_positive], dim=-1)

        return Scores.from_sequence_scores(probs, batch_indices=X.batch_indices)

    def save(self, path: Path | str) -> None:
        """Save the probe to disk.

        Args:
            path: Path to save the probe

        Raises:
            RuntimeError: If probe not fitted
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted probe")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "n_heads": self.n_heads,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "dropout": self.dropout,
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
    def load(cls, path: Path | str, device: str | None = None) -> "MultiMax":
        """Load a probe from disk.

        Args:
            path: Path to load the probe from
            device: Device to load onto (auto-detected if None)

        Returns:
            Loaded probe instance
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu")

        probe = cls(
            n_heads=state["n_heads"],
            mlp_hidden_dim=state["mlp_hidden_dim"],
            dropout=state.get("dropout", 0.1),
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
            f"MultiMax(n_heads={self.n_heads}, "
            f"mlp_hidden_dim={self.mlp_hidden_dim}, {fitted_str})"
        )
