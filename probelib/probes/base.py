"""Base class for all probes with unified interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union

import torch

from ..processing.activations import ActivationIterator, Activations


class BaseProbe(ABC):
    """Shared lifecycle for probes used throughout probelib.

    Probes load activations for a *single* model layer, optionally aggregate over
    tokens, and perform binary classification. Subclasses implement ``fit`` and
    scoring logic but inherit a consistent interface for device placement,
    streaming ``partial_fit`` updates, and token/sequence aggregation.

    Key design decisions:
    - Only single-layer activations today to keep training cheap and predictable.
    - Binary labels (0/1) align with the benchmark datasets and metrics.
    - No direct ``nn.Module`` inheritance so probes can use either Torch or
      sklearn under the hood without extra mixins.
    - Probes expect :class:`Activations` objects supplied by the processing
      pipeline; helpers such as ``train_probes`` make sure the right layer is
      collected once and reused across probes.
    """

    def __init__(
        self,
        layer: int,
        sequence_aggregation: Literal["mean", "max", "last_token"] | None = None,
        score_aggregation: Literal["mean", "max", "last_token"] | None = None,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize base probe.

        Args:
            layer: Layer index to use activations from
            sequence_aggregation: Aggregate sequences BEFORE training (classic)
            score_aggregation: Train on tokens, aggregate AFTER prediction
            device: Device for computation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.layer = layer

        # Validate aggregation parameters are mutually exclusive
        if sequence_aggregation is not None and score_aggregation is not None:
            raise ValueError(
                "Cannot use both sequence_aggregation and score_aggregation"
            )

        # Validate aggregation methods
        valid_aggregations = {"mean", "max", "last_token"}
        if (
            sequence_aggregation is not None
            and sequence_aggregation not in valid_aggregations
        ):
            raise ValueError(
                f"Invalid sequence_aggregation method '{sequence_aggregation}'. "
                f"Must be one of {valid_aggregations}"
            )
        if (
            score_aggregation is not None
            and score_aggregation not in valid_aggregations
        ):
            raise ValueError(
                f"Invalid score_aggregation method '{score_aggregation}'. "
                f"Must be one of {valid_aggregations}"
            )

        self.sequence_aggregation = sequence_aggregation
        self.score_aggregation = score_aggregation

        # Set device - auto-detect if None
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state
        self.verbose = verbose

        # State variables
        self._fitted = False
        self._requires_grad = False  # Set by subclasses if they need gradients

    def _prepare_features(self, X: Activations) -> torch.Tensor:
        """
        Prepare features from activations based on probe configuration.

        Args:
            X: Activations object

        Returns:
            Feature tensor ready for probe
        """
        # Ensure we have the correct layer
        if self.layer not in X.layer_indices:
            raise ValueError(
                f"Layer {self.layer} not found in activations. "
                f"Available layers: {X.layer_indices}"
            )

        # Filter to just the layer we need
        if len(X.layer_indices) > 1:
            X = X.filter_layers([self.layer])

        if X.n_layers != 1:
            raise ValueError(f"Expected single layer after filtering, got {X.n_layers}")

        # Aggregate sequence features up front when ``sequence_aggregation`` is set;
        # otherwise keep token-level tensors and let the probe aggregate scores
        # later. Both paths populate ``self._tokens_per_sample`` so subclasses can
        # recover sequence boundaries during ``score_aggregation``.
        if self.sequence_aggregation is not None:
            features = X.aggregate(method=self.sequence_aggregation)
        else:
            features, self._tokens_per_sample = X.to_token_level()

        # Return features on their original device (don't force to self.device)
        # Each probe will handle device placement as needed
        return features

    def _aggregate_scores(
        self, scores: torch.Tensor, method: Literal["mean", "max", "last_token"]
    ) -> torch.Tensor:
        """
        Aggregate token-level scores to sequence-level predictions.

        Args:
            scores: Token-level scores [n_tokens] or [n_tokens, n_classes]
            method: Aggregation method

        Returns:
            Sequence-level scores [n_samples] or [n_samples, n_classes]
        """
        if not hasattr(self, "_tokens_per_sample"):
            raise RuntimeError(
                "_tokens_per_sample not set. This should be set during _prepare_features."
            )

        split_sizes = self._tokens_per_sample.tolist()

        aggregated = []
        offset = 0
        for size in split_sizes:
            if size == 0:
                if scores.dim() == 1:
                    zero_value = torch.zeros(
                        (), device=scores.device, dtype=scores.dtype
                    )
                else:
                    zero_value = torch.zeros(
                        scores.shape[1], device=scores.device, dtype=scores.dtype
                    )
                aggregated.append(zero_value)
                offset += size
                continue

            if scores.dim() == 1:
                sample_score = scores[offset : offset + size]
            else:
                sample_score = scores[offset : offset + size, :]

            if method == "mean":
                aggregated.append(sample_score.mean(dim=0))
            elif method == "max":
                aggregated.append(
                    sample_score.max(dim=0).values
                    if sample_score.dim() > 1
                    else sample_score.max()
                )
            elif method == "last_token":
                aggregated.append(sample_score[-1])
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            offset += size

        # Stack results
        if scores.dim() == 1:
            return torch.stack(aggregated)
        else:
            return torch.stack(aggregated, dim=0)

    def _prepare_labels(
        self, y: list | torch.Tensor, expand_for_tokens: bool = False
    ) -> torch.Tensor:
        """
        Convert labels to tensor and optionally expand for token-level training.

        Args:
            y: List of labels (Label enum or int)
            expand_for_tokens: If True, expand labels to match token count

        Returns:
            Label tensor
        """
        if isinstance(y, torch.Tensor):
            labels = y
        elif hasattr(y[0], "value"):
            # Handle Label enum
            labels = torch.tensor([label.value for label in y])
        else:
            labels = torch.tensor(y)

        # Validate binary classification - check that all labels are 0 or 1
        # (but we don't require both classes to be present in every batch)
        unique_labels = labels.unique()
        if not torch.all((unique_labels == 0) | (unique_labels == 1)):
            raise ValueError(
                f"Only binary classification is supported. "
                f"Expected labels in [0, 1], got {unique_labels.tolist()}"
            )

        # Expand for token-level if needed
        if expand_for_tokens and hasattr(self, "_tokens_per_sample"):
            labels = torch.repeat_interleave(
                labels.to(self.device), self._tokens_per_sample.to(self.device)
            )

        return labels.to(self.device)

    @abstractmethod
    def fit(
        self,
        X: Union[Activations, ActivationIterator],
        y: list | torch.Tensor,
    ) -> "BaseProbe":
        """
        Fit the probe on training data.

        Must be implemented by subclasses.
        Should handle both Activations and ActivationIterator.
        Must set self._fitted = True when complete.
        """
        pass

    @abstractmethod
    def partial_fit(
        self,
        X: Activations,
        y: list | torch.Tensor,
    ) -> "BaseProbe":
        """
        Incrementally fit the probe (for streaming).

        Must be implemented by subclasses.
        No classes parameter - always assumes [0, 1].
        Must set self._fitted = True.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Union[Activations, ActivationIterator]) -> torch.Tensor:
        """
        Predict class probabilities.

        Must be implemented by subclasses.
        Should return tensor with shape [n_samples, 2].
        """
        pass

    def predict(self, X: Union[Activations, ActivationIterator]) -> torch.Tensor:
        """
        Predict class labels.

        Returns:
            Tensor of predicted class labels
        """
        probs = self.predict_proba(X)
        return probs.argmax(dim=1)

    def score(
        self, X: Union[Activations, ActivationIterator], y: list | torch.Tensor
    ) -> float:
        """
        Compute accuracy score.

        Args:
            X: Activations or ActivationIterator
            y: True labels

        Returns:
            Accuracy as a float
        """
        preds = self.predict(X)
        y_true = self._prepare_labels(y, expand_for_tokens=False)
        return (preds == y_true).float().mean().item()

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """
        Save the probe to disk.

        Args:
            path: Path to save the probe
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str, device: str | None = None) -> "BaseProbe":
        """
        Load a probe from disk.

        Args:
            path: Path to load the probe from
            device: Device to load onto (None to use saved device)

        Returns:
            Loaded probe instance
        """
        pass

    @property
    def requires_grad(self) -> bool:
        """Whether this probe requires gradients for training."""
        return self._requires_grad

    def __repr__(self) -> str:
        """String representation of the probe."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        agg_str = ""
        if self.sequence_aggregation is not None:
            agg_str = f"sequence_aggregation='{self.sequence_aggregation}', "
        elif self.score_aggregation is not None:
            agg_str = f"score_aggregation='{self.score_aggregation}', "
        return f"{self.__class__.__name__}(layer={self.layer}, {agg_str}{fitted_str})"
