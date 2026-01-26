"""Base class for all probes with unified interface."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from ..config import DEFAULT_DEVICE, VERBOSE
from ..processing.activations import Activations
from ..processing.scores import Scores


class BaseProbe(ABC):
    """Base class for probes in probelab.

    Probes are classifiers that operate on Activations objects and return
    Scores objects. They adapt their behavior based on the dimensionality
    of the input activations:
    - If activations have SEQ axis: Train/predict on tokens, return token-level Scores
    - If activations don't have SEQ axis: Train/predict on sequences, return sequence-level Scores

    Preprocessing (layer selection, aggregation) is handled by Pipeline
    transformers, not by the probe itself.

    Args:
        device: Device for computation ('cuda', 'cpu', or None for auto-detect)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information

    Example:
        >>> # Probes should be used within a Pipeline
        >>> from probelab import Pipeline
        >>> from probelab.preprocessing import SelectLayer, Pool
        >>> from probelab.probes import Logistic
        >>>
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("pool", Pool(dim="sequence", method="mean")),
        ...     ("probe", Logistic()),
        ... ])
        >>> pipeline.fit(activations, labels)
        >>> predictions = pipeline.predict_proba(test_activations)
    """

    def __init__(
        self,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool | None = None,
    ):
        """Initialize base probe.

        Args:
            device: Device for computation. If None, uses DEFAULT_DEVICE from config
                   (default: "cuda" if available, else "cpu")
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information. If None, uses VERBOSE from config.
        """
        # Set device - use config default, then auto-detect
        if device is None:
            config_device = DEFAULT_DEVICE.get()
            # Validate device availability
            if config_device.startswith("cuda") and not torch.cuda.is_available():
                self.device = "cpu"
            else:
                self.device = config_device
        else:
            self.device = device

        self.random_state = random_state
        self.verbose = verbose if verbose is not None else VERBOSE.get()

        # Track fitting state
        self._fitted = False
        self._tokens_per_sample = None  # Set when training on tokens

    def _to_tensor(self, y: list | torch.Tensor) -> torch.Tensor:
        """Convert labels to tensor.

        Args:
            y: Labels as list of Label enums, list of ints, or Tensor

        Returns:
            Label tensor

        Raises:
            ValueError: If labels are not binary (0 or 1)
        """
        if isinstance(y, torch.Tensor):
            labels = y
        elif hasattr(y[0], "value"):
            # Handle Label enum
            labels = torch.tensor([label.value for label in y])
        else:
            labels = torch.tensor(y)

        # Validate binary classification
        unique_labels = labels.unique()
        if not torch.all((unique_labels == 0) | (unique_labels == 1)):
            raise ValueError(
                f"Only binary classification is supported. "
                f"Expected labels in {{0, 1}}, got {unique_labels.tolist()}"
            )

        return labels

    # Abstract methods that subclasses must implement
    @abstractmethod
    def fit(
        self,
        X: Activations,
        y: list | torch.Tensor,
    ) -> "BaseProbe":
        """Fit the probe on activations and labels.

        Probes should inspect X.axes to determine behavior:
        - If X.has_axis(Axis.SEQ): Train on individual tokens
        - Otherwise: Train on sequence-level features

        Args:
            X: Activations with any valid shape
            y: Labels [batch] or [batch, seq]

        Returns:
            self: The fitted probe instance

        Raises:
            ValueError: If X has unexpected axes (e.g., LAYER axis present)
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: Activations,
    ) -> Scores:
        """Predict class probabilities.

        Returns Scores object matching input dimensionality:
        - If X has SEQ axis: Returns token-level Scores [batch, seq, 2]
        - Otherwise: Returns sequence-level Scores [batch, 2]

        Post-processing (e.g., aggregating token scores) can be done
        by Pool transform in the pipeline.

        Args:
            X: Activations to predict on

        Returns:
            Scores object with predictions

        Raises:
            ValueError: If probe not fitted
        """
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save the probe to disk.

        Args:
            path: File path to save the probe
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str, device: str | None = None) -> "BaseProbe":
        """Load a probe from disk.

        Args:
            path: File path to load the probe from
            device: Device to load the probe onto

        Returns:
            Loaded probe instance
        """
        pass

    def __repr__(self) -> str:
        """String representation of the probe."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"
