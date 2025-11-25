"""sklearn-style Pipeline for composing preprocessing and probes."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .preprocessing.base import PreTransformer
    from .probes.base import BaseProbe
    from .processing.activations import ActivationIterator, Activations
    from .processing.scores import Scores


class Pipeline:
    """sklearn-style pipeline for probelab.

    A Pipeline chains together a sequence of transformers with a probe:
    - Pre-probe transforms: Activations → Activations (SelectLayer, Pool, Normalize)
    - BaseProbe: Activations → Scores (classification)
    - Post-probe transforms: Scores → Scores (Pool for score aggregation)

    Exactly one step must be a BaseProbe. Pool can appear before the probe
    (to aggregate activations) or after (to aggregate token-level scores).

    Args:
        steps: List of (name, transformer/probe) tuples

    Example:
        >>> # Pre-probe pooling (sequence-level training)
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("pool", Pool(dim="sequence", method="mean")),
        ...     ("probe", Logistic()),
        ... ])
        >>> pipeline.fit(acts_train, labels_train)
        >>> predictions = pipeline.predict_proba(acts_test)

    Example with post-probe pooling (token-level training):
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("probe", Logistic()),  # Returns token-level Scores
        ...     ("pool", Pool(dim="sequence", method="mean")),  # Aggregate to sequence
        ... ])
    """

    def __init__(
        self,
        steps: list[tuple[str, "PreTransformer | BaseProbe"]],
    ):
        """Initialize pipeline with steps.

        Args:
            steps: List of (name, transformer/probe) tuples.
                   Exactly one step must be a BaseProbe.

        Raises:
            ValueError: If pipeline is invalid
        """
        if len(steps) == 0:
            raise ValueError("Pipeline cannot be empty")

        self.steps = steps
        self._validate_and_split_steps()

    def _validate_and_split_steps(self):
        """Validate pipeline structure and split into pre/probe/post."""
        from .preprocessing.base import PreTransformer
        from .probes.base import BaseProbe

        # Find the probe
        probe_indices = [
            i for i, (name, step) in enumerate(self.steps) if isinstance(step, BaseProbe)
        ]

        if len(probe_indices) == 0:
            raise ValueError("Pipeline must contain exactly one BaseProbe")
        if len(probe_indices) > 1:
            probe_names = [self.steps[i][0] for i in probe_indices]
            raise ValueError(
                f"Pipeline must contain exactly one BaseProbe, found {len(probe_indices)}: "
                f"{probe_names}"
            )

        probe_idx = probe_indices[0]

        # Validate pre-probe steps are PreTransformers
        for i, (name, step) in enumerate(self.steps[:probe_idx]):
            if not isinstance(step, PreTransformer):
                raise ValueError(
                    f"Step '{name}' (index {i}) must be a PreTransformer "
                    f"(before probe), got {type(step).__name__}"
                )

        # Post-probe steps should also be PreTransformers (Pool works on both)
        for i, (name, step) in enumerate(self.steps[probe_idx + 1 :], start=probe_idx + 1):
            if not isinstance(step, PreTransformer):
                raise ValueError(
                    f"Step '{name}' (index {i}) must be a transform "
                    f"(after probe), got {type(step).__name__}"
                )

        # Store split steps for efficient access
        self._pre_steps = self.steps[:probe_idx]
        self._probe_name, self._probe = self.steps[probe_idx]
        self._post_steps = self.steps[probe_idx + 1 :]

    def fit(
        self,
        X: "Activations",
        y: torch.Tensor | list,
    ) -> "Pipeline":
        """Fit the pipeline on training data.

        Pre-transformers are fit and applied in sequence, then the probe
        is fit on the transformed activations.

        Args:
            X: Training activations
            y: Training labels (Tensor or list of Labels/ints)

        Returns:
            self: Fitted pipeline
        """
        X_transformed = X
        for name, transformer in self._pre_steps:
            X_transformed = transformer.fit_transform(X_transformed, y)

        self._probe.fit(X_transformed, y)

        return self

    def partial_fit(
        self,
        X: "Activations",
        y: torch.Tensor | list,
    ) -> "Pipeline":
        """Incrementally fit pipeline on a batch of data.

        For streaming/online learning. Call multiple times with different batches.
        Pre-transformers are applied (assumed stateless), then probe is incrementally
        fitted using its partial_fit() method.

        **Note**: Post-transforms (Pool on Scores) are not supported with streaming.

        Args:
            X: Batch of activations
            y: Batch of labels

        Returns:
            self: The pipeline instance

        Raises:
            NotImplementedError: If probe doesn't support partial_fit or if
                pipeline has post-transforms

        Example:
            >>> for batch_acts in activation_iterator:
            ...     batch_labels = labels[batch_acts.batch_indices]
            ...     pipeline.partial_fit(batch_acts, batch_labels)
        """
        # Check for post-transforms
        if self._post_steps:
            raise NotImplementedError(
                "Streaming (partial_fit) is not supported with post-probe transforms. "
                "Use Pool(dim='sequence') before the probe instead."
            )

        # Check probe supports partial_fit
        if not hasattr(self._probe, "partial_fit"):
            raise NotImplementedError(
                f"Probe {self._probe.__class__.__name__} doesn't support streaming. "
                f"Use fit() with complete data instead."
            )

        # Apply pre-transforms (stateless, just transform)
        X_transformed = X
        for name, transformer in self._pre_steps:
            # Most pre-transformers are stateless
            # If they have partial_fit, use it then transform; otherwise just transform
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(X_transformed, y)
            X_transformed = transformer.transform(X_transformed)

        # Partial fit probe
        self._probe.partial_fit(X_transformed, y)

        return self

    def fit_streaming(
        self,
        X: "ActivationIterator",
        y: torch.Tensor | list,
        verbose: bool = False,
    ) -> "Pipeline":
        """Fit pipeline using streaming/online learning (single pass).

        Iterates over activation batches once, calling partial_fit on each batch.
        Each batch is processed exactly once - no multi-epoch training.

        For multi-epoch training, use streaming=False (batch mode) where the
        probe's fit() method handles epochs internally.

        Args:
            X: ActivationIterator that yields activation batches
            y: Labels for entire dataset (indexed by batch_indices)
            verbose: Print progress

        Returns:
            self: Fitted pipeline

        Raises:
            NotImplementedError: If probe doesn't support partial_fit or
                pipeline has post-transforms

        Example:
            >>> acts_iter = pl.collect_activations(..., streaming=True)
            >>> pipeline.fit_streaming(acts_iter, labels)
        """
        from .logger import logger
        from .types import Label

        # Check for post-transforms
        if self._post_steps:
            raise NotImplementedError(
                "Streaming (fit_streaming) is not supported with post-probe transforms. "
                "Use Pool(dim='sequence') before the probe instead."
            )

        # Check probe supports partial_fit
        if not hasattr(self._probe, "partial_fit"):
            raise NotImplementedError(
                f"Probe {self._probe.__class__.__name__} doesn't support streaming. "
                f"Use fit() with complete data instead."
            )

        # Convert labels to tensor for indexing
        if isinstance(y, list):
            if y and isinstance(y[0], Label):
                y = torch.tensor([label.value for label in y])
            else:
                y = torch.tensor(y)

        # Single pass through the data - each batch is processed exactly once
        for batch_acts in X:
            # Get batch labels using batch_indices
            batch_indices = torch.as_tensor(batch_acts.batch_indices)
            batch_labels = y.index_select(0, batch_indices)

            # Call existing partial_fit (single gradient step per batch)
            self.partial_fit(batch_acts, batch_labels)

            if verbose:
                logger.info(f"Processed batch with {len(batch_indices)} samples")

        return self

    def predict_proba(self, X: "Activations") -> torch.Tensor:
        """Predict class probabilities.

        Applies all pre-transforms, runs probe prediction, then applies
        all post-transforms.

        Args:
            X: Activations to predict on

        Returns:
            Predicted probabilities [batch, 2]

        Raises:
            ValueError: If pipeline not fitted
        """
        if not self._probe._fitted:
            raise ValueError(
                "Pipeline must be fitted before predict. Call fit() first."
            )

        # Apply pre-transforms (Activations → Activations)
        X_transformed = X
        for name, transformer in self._pre_steps:
            X_transformed = transformer.transform(X_transformed)

        # Get predictions from probe (Activations → Scores)
        scores: "Scores" = self._probe.predict_proba(X_transformed)

        # Apply post-transforms (Scores → Scores)
        for name, transformer in self._post_steps:
            scores = transformer.transform(scores)

        return scores.scores

    def predict(self, X: "Activations") -> torch.Tensor:
        """Predict class labels.

        Args:
            X: Activations to predict on

        Returns:
            Predicted class labels (0 or 1) [batch]
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).long()

    def score(self, X: "Activations", y: torch.Tensor | list) -> float:
        """Compute accuracy score.

        Args:
            X: Activations to predict on
            y: True labels

        Returns:
            Accuracy as a float
        """
        predictions = self.predict(X)

        # Convert labels to tensor if needed
        if isinstance(y, list):
            from .types import Label

            if y and isinstance(y[0], Label):
                y_tensor = torch.tensor([label.value for label in y])
            else:
                y_tensor = torch.tensor(y)
        else:
            y_tensor = y

        return (predictions == y_tensor.to(predictions.device)).float().mean().item()

    def __getitem__(self, key: str | int):
        """Access pipeline steps by name or index.

        Args:
            key: Step name (str) or index (int)

        Returns:
            The transformer or probe at the specified step

        Raises:
            KeyError: If step name not found
            IndexError: If index out of range

        Examples:
            >>> pipeline["select_layer"]  # By name
            >>> pipeline[0]  # By index
        """
        if isinstance(key, str):
            for name, step in self.steps:
                if name == key:
                    return step
            raise KeyError(f"Step '{key}' not found in pipeline")
        elif isinstance(key, int):
            return self.steps[key][1]
        else:
            raise TypeError(f"Key must be str or int, got {type(key).__name__}")

    def get_probe(self) -> "BaseProbe":
        """Get the probe step from the pipeline.

        Returns:
            The BaseProbe in this pipeline

        Example:
            >>> probe = pipeline.get_probe()
            >>> probe.save("probe.pt")
        """
        return self._probe

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_reprs = [f"  ('{name}', {step})" for name, step in self.steps]
        return "Pipeline([\n" + ",\n".join(step_reprs) + "\n])"
