"""
Unified high-level workflow functions for training and evaluating pipelines.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch

from ..datasets import DialogueDataset
from ..logger import logger
from ..metrics import auroc, get_metric_by_name, recall_at_fpr, with_bootstrap
from ..pipeline import Pipeline
from ..preprocessing.pre_transforms import SelectLayer, SelectLayers
from ..processing import collect_activations
from ..processing.activations import ActivationIterator, Activations
from ..types import Label

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..masks import MaskFunction

# Type aliases for clarity
PipelineInput = Pipeline | Mapping[str, Pipeline]
PredictionsOutput = torch.Tensor | Mapping[str, torch.Tensor]
MetricsDict = Mapping[str, Any]
MetricsOutput = MetricsDict | Mapping[str, MetricsDict]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _metric_display_name(metric_fn: Callable) -> str:
    """Create a consistent display name for metric functions/partials."""

    if hasattr(metric_fn, "__name__") and metric_fn.__name__ not in {None, "<lambda>"}:
        return metric_fn.__name__

    if isinstance(metric_fn, functools.partial):
        base = _metric_display_name(metric_fn.func)
        arg_parts = [repr(arg) for arg in getattr(metric_fn, "args", ())]
        kw_items = getattr(metric_fn, "keywords", {}) or {}
        kw_parts = [f"{key}={value}" for key, value in kw_items.items()]
        params = ", ".join(arg_parts + kw_parts)
        return f"{base}({params})" if params else base

    if hasattr(metric_fn, "__class__") and metric_fn.__class__.__name__ != "function":
        return metric_fn.__class__.__name__

    return repr(metric_fn)


def _extract_required_layers(pipelines: dict[str, Pipeline]) -> set[int]:
    """Extract all layers required by pipelines.

    Analyzes SelectLayer and SelectLayers transformers in pipelines
    to determine which layers need to be collected.

    Args:
        pipelines: Dict of name → Pipeline

    Returns:
        Set of layer indices to collect
    """
    required_layers = set()

    for pipeline in pipelines.values():
        for name, step in pipeline.steps:
            if isinstance(step, SelectLayer):
                required_layers.add(step.layer)
            elif isinstance(step, SelectLayers):
                required_layers.update(step.layers)

    return required_layers


def train_probes(
    pipelines: PipelineInput,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: DialogueDataset,
    *,
    mask: "MaskFunction" | None = None,
    batch_size: int = 32,
    streaming: bool = False,
    verbose: bool = True,
    **activation_kwargs: Any,
) -> None:
    """Train one or many pipelines while reusing a single activation pass.

    This function collects activations once and trains multiple pipelines
    in parallel, maximizing efficiency by avoiding repeated model forward passes.

    Args:
        pipelines: Single Pipeline or mapping name → Pipeline instance.
        model: Language model whose activations are collected.
        tokenizer: Tokenizer aligned with the model.
        dataset: DialogueDataset containing dialogues and labels.
        mask: Optional mask function for token selection.
        batch_size: Number of sequences per activation batch.
        streaming: Whether to force streaming activations (for large datasets).
        verbose: Toggle progress reporting.
        **activation_kwargs: Forwarded to :func:`collect_activations` for advanced
            control (e.g. hook_point, add_generation_prompt).

    Examples:
        >>> # Single pipeline
        >>> pipeline = pl.Pipeline([
        ...     ("select", pl.preprocessing.SelectLayer(16)),
        ...     ("agg", pl.preprocessing.AggregateSequences("mean")),
        ...     ("probe", pl.probes.Logistic()),
        ... ])
        >>> train_probes(pipeline, model, tokenizer, dataset)

        >>> # Multiple pipelines (parallel training, single collection)
        >>> pipelines = {
        ...     "mean": pl.Pipeline([...]),
        ...     "max": pl.Pipeline([...]),
        ... }
        >>> train_probes(pipelines, model, tokenizer, dataset)
    """
    # 1. Normalize inputs
    is_single_pipeline = isinstance(pipelines, Pipeline)
    pipelines_dict = {"_single": pipelines} if is_single_pipeline else pipelines

    # 2. Get labels from dataset
    labels = dataset.labels
    if isinstance(labels, list) and labels and isinstance(labels[0], Label):
        labels_tensor = torch.tensor([label.value for label in labels])
    else:
        labels_tensor = (
            torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        )

    # 3. Determine all required layers from pipelines
    required_layers = _extract_required_layers(pipelines_dict)

    if not required_layers:
        # No layer selection specified - collect all layers
        # This shouldn't happen in practice with well-formed pipelines
        raise ValueError(
            "Could not determine required layers from pipelines. "
            "Ensure pipelines include SelectLayer or SelectLayers transformers."
        )

    if verbose:
        logger.info(f"Collecting activations for layers: {sorted(required_layers)}")

    # 4. Collect activations once for all layers
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=sorted(required_layers),
        mask=mask,
        batch_size=batch_size,
        streaming=streaming,
        verbose=verbose,
        detach_activations=activation_kwargs.pop("detach_activations", True),
        **activation_kwargs,
    )

    # 5. Train each pipeline
    if isinstance(activations, ActivationIterator):
        raise NotImplementedError(
            "Streaming mode for train_probes is not yet implemented with pipelines. "
            "Use streaming=False or implement pipeline partial_fit support."
        )
    else:
        _train_pipelines_batch(pipelines_dict, activations, labels_tensor, verbose)


def _train_pipelines_batch(
    pipelines: dict[str, Pipeline],
    activations: Activations,
    labels: torch.Tensor,
    verbose: bool,
) -> None:
    """Train all pipelines using in-memory activations."""
    if verbose:
        logger.info(f"Training {len(pipelines)} pipeline(s)")

    for name, pipeline in pipelines.items():
        if verbose and len(pipelines) > 1:
            logger.info(f"  Training pipeline: {name}")

        # Each pipeline applies its own preprocessing then trains
        pipeline.fit(activations, labels)


def evaluate_probes(
    pipelines: PipelineInput,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: DialogueDataset,
    *,
    mask: "MaskFunction" | None = None,
    batch_size: int = 32,
    streaming: bool = False,
    metrics: list[Callable | str] | None = None,
    bootstrap: bool | dict[str, Any] | None = None,
    verbose: bool = True,
    **activation_kwargs: Any,
) -> tuple[PredictionsOutput, MetricsOutput]:
    """Evaluate one or many pipelines.

    Args:
        pipelines: Single Pipeline or dict mapping names to Pipelines
        model: Language model to extract activations from
        tokenizer: Tokenizer for the model
        dataset: DialogueDataset containing dialogues and labels
        mask: Optional mask function for token selection
        batch_size: Batch size for activation collection
        streaming: Whether to use streaming mode for large datasets
        metrics: List of metric functions or names (defaults to standard set)
        bootstrap: ``None`` (default) disables bootstrap; pass ``True`` to apply
            default bootstrap settings or a dict of keyword arguments accepted by
            :func:`probelib.metrics.with_bootstrap` for customisation.
        verbose: Whether to show progress bars
        **activation_kwargs: Additional args passed to collect_activations

    Returns:
        Tuple of (predictions, metrics) where:
        - predictions: Tensor or dict of tensors with predicted probabilities
        - metrics: Dict or dict of dicts with computed metrics

    Examples:
        >>> # Single pipeline
        >>> predictions, metrics = evaluate_probes(
        ...     pipeline, model, tokenizer, test_dataset
        ... )
        >>> print(f"AUROC: {metrics['auroc']:.3f}")

        >>> # Multiple pipelines
        >>> predictions, metrics = evaluate_probes(
        ...     pipelines, model, tokenizer, test_dataset,
        ...     metrics=["auroc", "balanced_accuracy"]
        ... )
        >>> print(f"Mean pipeline AUROC: {metrics['mean']['auroc']:.3f}")
    """
    # 1. Normalize inputs
    is_single_pipeline = isinstance(pipelines, Pipeline)
    pipelines_dict = {"_single": pipelines} if is_single_pipeline else pipelines

    # 2. Get labels from dataset
    labels = dataset.labels
    if isinstance(labels, list) and labels and isinstance(labels[0], Label):
        labels_tensor = torch.tensor([label.value for label in labels])
    else:
        labels_tensor = (
            torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        )

    # 3. Set default metrics
    if metrics is None:
        metrics = [
            auroc,
            functools.partial(recall_at_fpr, fpr=0.001),  # recall@0.1%
            functools.partial(recall_at_fpr, fpr=0.01),  # recall@1%
        ]
    else:
        # Convert any string metrics to functions
        converted_metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                converted_metrics.append(get_metric_by_name(metric))
            else:
                converted_metrics.append(metric)
        metrics = converted_metrics

    # Normalise bootstrap configuration
    if isinstance(bootstrap, bool):
        bootstrap_kwargs: dict[str, Any] | None = {} if bootstrap else None
    else:
        bootstrap_kwargs = bootstrap

    # 4. Determine required layers
    required_layers = _extract_required_layers(pipelines_dict)

    if not required_layers:
        raise ValueError(
            "Could not determine required layers from pipelines. "
            "Ensure pipelines include SelectLayer or SelectLayers transformers."
        )

    if verbose:
        logger.info(f"Collecting activations for layers: {sorted(required_layers)}")

    # 5. Collect activations
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=sorted(required_layers),
        mask=mask,
        batch_size=batch_size,
        streaming=streaming,
        verbose=verbose,
        detach_activations=activation_kwargs.pop("detach_activations", True),
        **activation_kwargs,
    )

    # 6. Evaluate
    if isinstance(activations, ActivationIterator):
        raise NotImplementedError(
            "Streaming mode for evaluate_probes is not yet implemented with pipelines."
        )
    else:
        all_predictions, all_metrics = _evaluate_pipelines_batch(
            pipelines_dict, activations, labels_tensor, metrics, bootstrap_kwargs
        )

    # 7. Return in same format as input
    if is_single_pipeline:
        return all_predictions["_single"], all_metrics["_single"]
    else:
        return all_predictions, all_metrics


def _evaluate_pipelines_batch(
    pipelines: dict[str, Pipeline],
    activations: Activations,
    labels: torch.Tensor,
    metrics: list[Callable],
    bootstrap_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, MetricsDict]]:
    """Evaluate all pipelines using in-memory activations."""

    all_predictions = {}
    all_metrics = {}

    for name, pipeline in pipelines.items():
        # Get predictions (pipeline returns [batch, 2])
        probs = pipeline.predict_proba(activations)
        preds = probs[:, 1]  # Positive class probabilities

        all_predictions[name] = preds

        # Compute metrics
        pipeline_metrics = {}

        # Convert to numpy for metrics
        y_true = (
            labels.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor)
            else labels
        )
        y_pred = (
            preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        )

        for metric_fn in metrics:
            metric_name = _metric_display_name(metric_fn)
            metric_callable = metric_fn
            if bootstrap_kwargs is not None and not getattr(
                metric_fn, "_probelib_bootstrap", False
            ):
                metric_callable = with_bootstrap(**bootstrap_kwargs)(metric_fn)

            # Compute metric (optionally bootstrapped)
            result = metric_callable(y_true, y_pred)
            pipeline_metrics[metric_name] = result

        all_metrics[name] = pipeline_metrics

    return all_predictions, all_metrics
