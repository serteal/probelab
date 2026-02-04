"""Base classes for transforms.

Provides type-safe base classes for transforms that operate on different data types:
- ActivationTransform: Activations → Activations (pre-probe transforms)
- ScoreTransform: Scores → Scores (post-probe transforms)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..processing.activations import Activations
    from ..processing.scores import Scores


class ActivationTransform(ABC):
    """Base class for transforms that operate on Activations.

    ActivationTransforms are used before the probe in a Pipeline. They transform
    activation data (layer selection, pooling, normalization) while preserving
    or modifying the axis structure.

    Type signature: Activations → Activations

    Examples:
        - SelectLayer: Selects single layer, removes LAYER axis
        - SelectLayers: Selects multiple layers, keeps LAYER axis
        - Pool: Reduces layer or sequence dimension
        - Normalize: Standardizes features
    """

    def fit(self, X: "Activations", y=None) -> "ActivationTransform":
        """Fit transformer (optional).

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
            X: Input activations

        Returns:
            Transformed activations
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


class ScoreTransform(ABC):
    """Base class for transforms that operate on Scores.

    ScoreTransforms are used after the probe in a Pipeline. They transform
    prediction scores (aggregation, calibration) typically reducing token-level
    scores to sequence-level scores.

    Type signature: Scores → Scores

    Examples:
        - Pool: Aggregates token scores to sequence level
        - EMAPool: Exponential moving average aggregation
        - RollingPool: Rolling window aggregation
    """

    def fit(self, X: "Scores", y=None) -> "ScoreTransform":
        """Fit transformer (optional).

        Most transforms are stateless, but some (like calibration) need to
        compute statistics from training data.

        Args:
            X: Training scores
            y: Labels (unused, for sklearn compatibility)

        Returns:
            self: Fitted transformer
        """
        return self

    @abstractmethod
    def transform(self, X: "Scores") -> "Scores":
        """Transform scores.

        Args:
            X: Input scores

        Returns:
            Transformed scores
        """
        pass

    def fit_transform(self, X: "Scores", y=None) -> "Scores":
        """Fit and transform in one step.

        Args:
            X: Training scores
            y: Labels (unused, for sklearn compatibility)

        Returns:
            Transformed scores
        """
        return self.fit(X, y).transform(X)
