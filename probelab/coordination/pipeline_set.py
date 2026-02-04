"""PipelineSet for coordinated multi-pipeline training.

This module provides the PipelineSet class for training multiple pipelines
on the same activations with step fusion to avoid redundant computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..pipeline import Pipeline
    from ..processing.activations import Activations
    from ..types import Label

from .graph import ExecutionGraph


class PipelineSet:
    """Coordinate training of multiple pipelines with shared transforms.

    PipelineSet analyzes multiple pipelines to detect shared transform
    prefixes, then trains them efficiently by computing shared transforms
    only once.

    Example:
        >>> # Create pipelines with different configurations
        >>> pipelines = {
        ...     "layer_16_mean": Pipeline([
        ...         ("select", pre.SelectLayer(16)),
        ...         ("pool", pre.Pool(dim="sequence", method="mean")),
        ...         ("probe", Logistic()),
        ...     ]),
        ...     "layer_16_max": Pipeline([
        ...         ("select", pre.SelectLayer(16)),
        ...         ("pool", pre.Pool(dim="sequence", method="max")),
        ...         ("probe", Logistic()),
        ...     ]),
        ...     "layer_20_mean": Pipeline([
        ...         ("select", pre.SelectLayer(20)),
        ...         ("pool", pre.Pool(dim="sequence", method="mean")),
        ...         ("probe", Logistic()),
        ...     ]),
        ... }
        >>>
        >>> # Create PipelineSet - detects shared SelectLayer(16) between first two
        >>> pipeline_set = PipelineSet(pipelines)
        >>> print(pipeline_set.graph.summary())
        >>>
        >>> # Train all pipelines with fused transforms
        >>> pipeline_set.fit(activations, labels)
        >>>
        >>> # Get predictions from all pipelines
        >>> results = pipeline_set.predict(activations)
    """

    def __init__(
        self,
        pipelines: dict[str, "Pipeline"] | list["Pipeline"],
    ):
        """Initialize PipelineSet with pipelines.

        Args:
            pipelines: Either a dict mapping names to pipelines,
                      or a list of pipelines (auto-named pipeline_0, etc.)
        """
        if isinstance(pipelines, list):
            self._pipelines = {f"pipeline_{i}": p for i, p in enumerate(pipelines)}
        else:
            self._pipelines = pipelines

        self.graph = ExecutionGraph.from_pipelines(self._pipelines)
        self._fitted = False
        self._node_cache: dict[int, Any] = {}

    @property
    def pipelines(self) -> dict[str, "Pipeline"]:
        """Get the pipelines in this set."""
        return self._pipelines

    def _execute_fused_pre_transforms(
        self,
        X: "Activations",
        y: torch.Tensor | list | None = None,
        fit: bool = False,
    ) -> dict[int, "Activations"]:
        """Execute pre-transforms with fusion.

        Walks the graph in topological order, computing each transform once
        and caching the result. Transforms that are shared across multiple
        pipelines are computed only once.

        Args:
            X: Input activations
            y: Labels (needed for fit=True)
            fit: Whether to fit transforms (True during training)

        Returns:
            Dict mapping node IDs to their output activations
        """
        from ..transforms.base import ActivationTransform

        cache: dict[int, "Activations"] = {}
        execution_order = self.graph.get_execution_order()

        for node_id in execution_order:
            node = self.graph.nodes[node_id]

            # Skip non-pre-transforms
            if not isinstance(node.step, ActivationTransform):
                continue

            # Determine input for this node
            # Find parent node(s) in the graph
            parent_id = None
            for pid, children in self.graph.edges.items():
                if node_id in children:
                    parent_id = pid
                    break

            if parent_id is None:
                # Root node - use original activations
                input_acts = X
            else:
                # Use cached output from parent
                input_acts = cache[parent_id]

            # Apply transform (fit_transform if fitting, else transform)
            if fit:
                output_acts = node.step.fit_transform(input_acts, y)
            else:
                output_acts = node.step.transform(input_acts)

            cache[node_id] = output_acts

        return cache

    def _get_probe_input(
        self,
        pipeline_name: str,
        pre_transform_cache: dict[int, "Activations"],
        X: "Activations",
    ) -> "Activations":
        """Get the input activations for a specific pipeline's probe.

        Args:
            pipeline_name: Name of the pipeline
            pre_transform_cache: Cache of pre-transform outputs
            X: Original input (used if no pre-transforms)

        Returns:
            Activations ready for probe input
        """
        pipeline = self._pipelines[pipeline_name]
        path = self.graph.pipeline_paths[pipeline_name]

        # Find the last pre-transform node before the probe
        probe_idx = None
        for i, node_id in enumerate(path):
            if self.graph.nodes[node_id].is_probe:
                probe_idx = i
                break

        if probe_idx is None:
            raise ValueError(f"No probe found in pipeline {pipeline_name}")

        if probe_idx == 0:
            # No pre-transforms, use original input
            return pipeline._auto_select_single_layer(X)
        else:
            # Get output of last pre-transform
            last_pre_node_id = path[probe_idx - 1]
            return pre_transform_cache[last_pre_node_id]

    def fit(
        self,
        X: "Activations",
        y: torch.Tensor | list,
    ) -> "PipelineSet":
        """Fit all pipelines with fused execution.

        Shared transforms are computed once and reused across pipelines.

        Args:
            X: Training activations
            y: Training labels

        Returns:
            self: Fitted PipelineSet
        """
        # Execute all pre-transforms with fusion
        pre_transform_cache = self._execute_fused_pre_transforms(X, y, fit=True)

        # Fit each probe with its appropriate input
        for pipeline_name, pipeline in self._pipelines.items():
            probe_input = self._get_probe_input(pipeline_name, pre_transform_cache, X)
            pipeline._probe.fit(probe_input, y)

        self._fitted = True
        return self

    def predict(
        self,
        X: "Activations",
    ) -> dict[str, torch.Tensor]:
        """Get predictions from all pipelines.

        Args:
            X: Activations to predict on

        Returns:
            Dict mapping pipeline names to prediction tensors [batch, 2]
        """
        if not self._fitted:
            raise ValueError("PipelineSet must be fitted before predict")

        from ..processing.scores import Scores
        from ..transforms.base import ScoreTransform

        # Execute all pre-transforms with fusion
        pre_transform_cache = self._execute_fused_pre_transforms(X, fit=False)

        results: dict[str, torch.Tensor] = {}

        for pipeline_name, pipeline in self._pipelines.items():
            # Get probe input
            probe_input = self._get_probe_input(pipeline_name, pre_transform_cache, X)

            # Get probe predictions
            scores: Scores = pipeline._probe.predict(probe_input)

            # Apply post-transforms
            for step_name, step in pipeline._post_steps:
                if isinstance(step, ScoreTransform):
                    scores = step.transform(scores)

            results[pipeline_name] = scores.scores

        return results

    def score(
        self,
        X: "Activations",
        y: torch.Tensor | list,
    ) -> dict[str, float]:
        """Compute accuracy scores for all pipelines.

        Args:
            X: Activations to score on
            y: True labels

        Returns:
            Dict mapping pipeline names to accuracy scores
        """
        from ..types import Label

        predictions = self.predict(X)

        # Convert labels to tensor if needed
        if isinstance(y, list):
            if y and isinstance(y[0], Label):
                y_tensor = torch.tensor([label.value for label in y])
            else:
                y_tensor = torch.tensor(y)
        else:
            y_tensor = y

        scores = {}
        for name, probs in predictions.items():
            preds = probs.argmax(dim=1)
            scores[name] = (preds == y_tensor.to(preds.device)).float().mean().item()

        return scores

    def __getitem__(self, key: str) -> "Pipeline":
        """Access a pipeline by name.

        Args:
            key: Pipeline name

        Returns:
            The Pipeline object
        """
        return self._pipelines[key]

    def __len__(self) -> int:
        """Number of pipelines in this set."""
        return len(self._pipelines)

    def __repr__(self) -> str:
        """String representation."""
        shared_depth = self.graph.get_shared_prefix_depth()
        return (
            f"PipelineSet(pipelines={list(self._pipelines.keys())}, "
            f"shared_prefix_depth={shared_depth}, "
            f"fitted={self._fitted})"
        )
