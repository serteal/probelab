"""Base class for probes."""

import copy
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ..activations import Activations


class BaseProbe(ABC):
    """Base class for probes.

    Probes are classifiers that operate on Activations and return probability tensors.
    They adapt based on input dimensionality:
    - If activations have SEQ axis: Train/predict on tokens
    - If no SEQ axis: Train/predict on sequences

    Two interfaces:
    - probe(x): Differentiable forward pass on raw tensors, returns logits
    - probe.predict(X): Convenience method for Activations, returns probabilities

    Args:
        device: Device to use. If None, auto-detects from input in fit().
        seed: Random seed for reproducibility. If None, no seeding is done.
        optimizer_fn: Factory ``fn(params) -> optimizer``. If None, uses AdamW.
        scheduler_fn: Factory ``fn(optimizer) -> scheduler``. If None, no scheduler.
        cast: Dtype policy. ``None`` preserves input dtype (default).
            ``"float32"``/``"float16"``/``"bfloat16"`` forces that dtype.
    """

    _CAST_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        device: str | None = None,
        seed: int | None = None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        cast: str | None = None,
    ):
        self.device = device
        self.seed = seed
        self._optimizer_fn = optimizer_fn
        self._scheduler_fn = scheduler_fn
        self.cast = cast
        self._training_dtype: torch.dtype | None = None
        self._optimizer = None
        self._scheduler = None
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_state = None

    def _resolve_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        """Resolve working dtype from cast policy and input dtype."""
        if self.cast is not None:
            return self._CAST_MAP[self.cast]
        return input_dtype

    def _seed_everything(self) -> torch.Generator | None:
        """Seed all RNGs for reproducibility.

        Sets global seeds (torch, CUDA, numpy, random) so that weight
        initialisation and dropout are deterministic.  Returns a local
        ``torch.Generator`` that probes can pass to ``randperm`` and
        ``DataLoader`` for deterministic shuffling without further
        mutating global state.  Returns ``None`` when no seed is set.
        """
        if self.seed is None:
            return None
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

    def _make_optimizer(self, params, **defaults):
        """Create optimizer from factory or fall back to AdamW with defaults."""
        if self._optimizer_fn is not None:
            return self._optimizer_fn(params)
        fused = isinstance(self.device, str) and self.device.startswith("cuda")
        return AdamW(params, fused=fused, **defaults)

    def _make_scheduler(self, optimizer):
        """Create scheduler from factory, or return None."""
        if self._scheduler_fn is not None:
            return self._scheduler_fn(optimizer)
        return None

    # ------------------------------------------------------------------
    # Composable training interface — used by multi-probe sweep training
    # ------------------------------------------------------------------

    def _create_network(self, d_model: int) -> None:
        """Set ``self.net`` to a fresh nn.Module. Called after seeding.

        Override in seq probe subclasses (Attention, MultiMax, GatedBipolar).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _create_network"
        )

    def setup_training(self, d_model: int, working_dtype: torch.dtype) -> torch.Generator | None:
        """Create network, optimizer, scheduler; reset early-stopping state.

        Calls ``_seed_everything()`` first so weight init is deterministic,
        then ``_create_network(d_model)`` (subclass hook).

        Returns the seeded Generator (or ``None``).
        """
        self._training_dtype = working_dtype
        g = self._seed_everything()
        self._create_network(d_model)
        self.net.to(self.device, dtype=working_dtype)
        self._optimizer = self._make_optimizer(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        self._scheduler = self._make_scheduler(self._optimizer)
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_state = None
        return g

    def train_on_batch(
        self,
        batch_seq: torch.Tensor,
        batch_mask: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> None:
        """One gradient step on a pre-loaded batch (already on device)."""
        self._optimizer.zero_grad()
        out = self.net(batch_seq, batch_mask)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.binary_cross_entropy_with_logits(logits, batch_y)
        loss = loss + self._regularization_loss()
        loss.backward()
        self._optimizer.step()

    def _regularization_loss(self) -> float:
        """Override in subclasses with custom regularization (e.g. GatedBipolar)."""
        return 0.0

    def check_val(self, val_loss: float) -> bool:
        """Update early-stopping state after a validation check.

        Also steps the scheduler if one exists.
        Returns ``True`` when training should stop.
        """
        if self._scheduler is not None:
            self._scheduler.step()
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            if self.net is not None:
                self._best_state = copy.deepcopy(self.net.state_dict())
        else:
            self._patience_counter += 1
        return self._patience_counter >= self.patience

    def restore_best(self) -> None:
        """Restore network weights to the best validation checkpoint, if available."""
        if self._best_state is not None and self.net is not None:
            self.net.load_state_dict(self._best_state)

    def should_validate_at(self, epoch: int) -> bool:
        """Whether to run validation at this epoch. Default: every epoch."""
        return True

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Whether the probe has been fitted. Each probe defines its own check."""
        ...

    @abstractmethod
    def fit(self, X: Activations, y: list | torch.Tensor) -> "BaseProbe":
        """Fit probe on activations and labels."""
        ...

    @abstractmethod
    def predict(self, X: Activations) -> torch.Tensor:
        """Predict probabilities from Activations."""
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save probe to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "BaseProbe":
        """Load probe from disk."""
        ...

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before predict")

    def _to_labels(self, y) -> torch.Tensor:
        """Convert labels list to tensor."""
        if isinstance(y, torch.Tensor):
            return y
        return torch.tensor([l.value if hasattr(l, "value") else l for l in y])

    @staticmethod
    def _get_seq_lengths(X) -> torch.Tensor:
        """Get per-sample sequence lengths from Activations or compatible object."""
        if hasattr(X, "offsets") and X.offsets is not None:
            return X.offsets[1:] - X.offsets[:-1]
        # _SubsetActivations: compute from parent offsets
        if hasattr(X, "_parent") and hasattr(X, "_indices"):
            po = X._parent.offsets
            idx = torch.tensor(X._indices, dtype=torch.long)
            return po[idx + 1] - po[idx]
        raise ValueError("Cannot determine sequence lengths from input")

    @staticmethod
    def _length_sorted_batches(
        sample_indices: list[int],
        seq_lengths: torch.Tensor,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Group sample indices into length-sorted batches.

        Sorts samples by descending sequence length so each minibatch only
        pads to its own local max, then shuffles the batch order for
        stochasticity.

        Args:
            sample_indices: List of sample indices to batch.
            seq_lengths: Per-sample lengths (indexed by sample index).
            batch_size: Max samples per batch.
            generator: Optional RNG for batch-order shuffling.

        Returns:
            List of index-lists, each of length ≤ batch_size.
        """
        # Sort by descending length
        decorated = sorted(sample_indices, key=lambda i: -int(seq_lengths[i]))
        # Chunk into batches
        batches = [
            decorated[i : i + batch_size]
            for i in range(0, len(decorated), batch_size)
        ]
        # Shuffle batch order (not within batches — preserves length grouping)
        perm = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[p] for p in perm]

    @staticmethod
    def _minibatch_forward(
        net,
        X,
        indices: list[int],
        device: str,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        """Run net forward on X in minibatches, return concatenated logits.

        Pads each minibatch independently to avoid materializing the full
        padded tensor.
        """
        all_logits: list[torch.Tensor] = []
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_seq, batch_mask = X.pad_batch(batch_idx)
            batch_seq = batch_seq.to(device, dtype=dtype)
            batch_mask = batch_mask.to(device)
            out = net(batch_seq, batch_mask)
            # Some nets return (logits, extras), others return logits
            logits = out[0] if isinstance(out, tuple) else out
            all_logits.append(logits)
            del batch_seq, batch_mask
        return torch.cat(all_logits, dim=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.fitted})"
