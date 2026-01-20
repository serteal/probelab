"""sklearn-style Pipeline for composing preprocessing and probes."""

from typing import TYPE_CHECKING

import torch

from .processing.activations import Axis

if TYPE_CHECKING:
    from .preprocessing.base import PreTransformer
    from .probes.base import BaseProbe
    from .processing.activations import Activations
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

    def _has_layer_handling(self) -> bool:
        """Check if pipeline has explicit layer handling (SelectLayer, SelectLayers, or Pool on layer)."""
        from .preprocessing.pre_transforms import Pool, SelectLayer, SelectLayers

        for _, step in self._pre_steps:
            if isinstance(step, (SelectLayer, SelectLayers)):
                return True
            if isinstance(step, Pool) and step.dim == "layer":
                return True
        return False

    def _auto_select_single_layer(self, X: "Activations") -> "Activations":
        """Auto-select single layer if no explicit layer handling in pipeline.

        If activations have exactly one layer and the pipeline has no SelectLayer,
        SelectLayers, or Pool(dim="layer"), automatically select that layer.

        Args:
            X: Input activations

        Returns:
            Activations with LAYER axis removed if single layer, otherwise unchanged

        Raises:
            ValueError: If multiple layers present without explicit handling
        """
        if not X.has_axis(Axis.LAYER):
            return X

        if self._has_layer_handling():
            # Pipeline has explicit layer handling, don't auto-select
            return X

        if X.n_layers == 1:
            # Single layer - auto-select it
            return X.select(layer=X.layer_indices[0])
        else:
            # Multiple layers without handling - error with helpful message
            raise ValueError(
                f"Activations contain {X.n_layers} layers {X.layer_indices} "
                f"but pipeline has no layer handling.\n\n"
                f"Options:\n"
                f"  1. Collect single layer: collect_activations(..., layers=[{X.layer_indices[0]}])\n"
                f"  2. Add SelectLayer: Pipeline([SelectLayer({X.layer_indices[0]}), ...])\n"
                f"  3. Pool layers: Pipeline([Pool(dim='layer', method='mean'), ...])"
            )

    def fit(
        self,
        X: "Activations",
        y: torch.Tensor | list,
    ) -> "Pipeline":
        """Fit the pipeline on training data.

        Pre-transformers are fit and applied in sequence, then the probe
        is fit on the transformed activations.

        If activations have exactly one layer and the pipeline has no explicit
        layer handling (SelectLayer, SelectLayers, or Pool(dim="layer")), the
        single layer is automatically selected.

        Args:
            X: Training activations
            y: Training labels (Tensor or list of Labels/ints)

        Returns:
            self: Fitted pipeline

        Raises:
            ValueError: If activations have multiple layers without explicit handling
        """
        # Auto-select single layer if no explicit layer handling
        X_transformed = self._auto_select_single_layer(X)

        for name, transformer in self._pre_steps:
            X_transformed = transformer.fit_transform(X_transformed, y)

        self._probe.fit(X_transformed, y)

        return self

    def predict_proba(self, X: "Activations") -> torch.Tensor:
        """Predict class probabilities.

        Applies all pre-transforms, runs probe prediction, then applies
        all post-transforms.

        If activations have exactly one layer and the pipeline has no explicit
        layer handling (SelectLayer, SelectLayers, or Pool(dim="layer")), the
        single layer is automatically selected.

        Args:
            X: Activations to predict on

        Returns:
            Predicted probabilities [batch, 2]

        Raises:
            ValueError: If pipeline not fitted or multiple layers without handling
        """
        if not self._probe._fitted:
            raise ValueError(
                "Pipeline must be fitted before predict. Call fit() first."
            )

        # Auto-select single layer if no explicit layer handling
        X_transformed = self._auto_select_single_layer(X)

        # Apply pre-transforms (Activations → Activations)
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
