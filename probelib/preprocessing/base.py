"""Base classes for preprocessing transformers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ..processing.activations import Activations


class PreTransformer(ABC):
    """Base class for pre-probe transformers.

    PreTransformers operate on Activations objects, transforming them
    before they reach the probe. They preserve the Activations structure
    with updated metadata.

    Examples:
        - SelectLayer: Remove LAYER axis
        - AggregateSequences: Remove SEQ axis
        - Normalize: Keep shape, normalize values
    """

    def fit(self, X: "Activations", y=None) -> "PreTransformer":
        """Fit transformer on activations (optional).

        Most transforms are stateless, but some (like Normalize) need to
        compute statistics from training data.

        Args:
            X: Training activations
            y: Labels (unused, for sklearn compatibility)

        Returns:
            self: Fitted transformer
        """
        return self

    @abstractmethod
    def transform(self, X: "Activations") -> "Activations":
        """Transform activations.

        Args:
            X: Input activations with metadata

        Returns:
            Transformed activations with updated metadata
        """
        pass

    def fit_transform(self, X: "Activations", y=None) -> "Activations":
        """Fit and transform in one step.

        Args:
            X: Training activations
            y: Labels (unused, for sklearn compatibility)

        Returns:
            Transformed activations
        """
        return self.fit(X, y).transform(X)


class PostTransformer(ABC):
    """Base class for post-probe transformers.

    PostTransformers operate on score tensors produced by probes,
    transforming them after prediction but before final output.

    Examples:
        - AggregateTokenScores: [n_tokens, 2] → [batch, 2]
        - ThresholdScores: Apply threshold to scores
        - CalibrateScores: Calibrate probability scores
    """

    def fit(self, X: "torch.Tensor", y=None) -> "PostTransformer":
        """Fit transformer on scores (optional).

        Args:
            X: Training scores
            y: Labels (unused, for sklearn compatibility)

        Returns:
            self: Fitted transformer
        """
        return self

    @abstractmethod
    def transform(self, X: "torch.Tensor") -> "torch.Tensor":
        """Transform scores.

        Args:
            X: Input scores (e.g., [n_tokens, 2] or [batch, 2])

        Returns:
            Transformed scores
        """
        pass

    def fit_transform(self, X: "torch.Tensor", y=None) -> "torch.Tensor":
        """Fit and transform in one step.

        Args:
            X: Training scores
            y: Labels (unused, for sklearn compatibility)

        Returns:
            Transformed scores
        """
        return self.fit(X, y).transform(X)
