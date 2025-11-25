"""Base classes for preprocessing transformers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..processing.activations import Activations
    from ..processing.scores import Scores


class PreTransformer(ABC):
    """Base class for transformers in pipelines.

    PreTransformers can operate on both Activations objects (before the probe)
    and Scores objects (after the probe). The Pool class is a good example
    of a transformer that works on both.

    Examples:
        - SelectLayer: Activations → Activations (remove LAYER axis)
        - Pool: Activations → Activations OR Scores → Scores
        - Normalize: Activations → Activations (normalize values)
    """

    def fit(
        self, X: Union["Activations", "Scores"], y=None
    ) -> "PreTransformer":
        """Fit transformer (optional).

        Most transforms are stateless, but some (like Normalize) need to
        compute statistics from training data.

        Args:
            X: Training data (Activations or Scores)
            y: Labels (unused, for sklearn compatibility)

        Returns:
            self: Fitted transformer
        """
        return self

    @abstractmethod
    def transform(
        self, X: Union["Activations", "Scores"]
    ) -> Union["Activations", "Scores"]:
        """Transform data.

        Args:
            X: Input data (Activations or Scores)

        Returns:
            Transformed data
        """
        pass

    def fit_transform(
        self, X: Union["Activations", "Scores"], y=None
    ) -> Union["Activations", "Scores"]:
        """Fit and transform in one step.

        Args:
            X: Training data
            y: Labels (unused, for sklearn compatibility)

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
