"""
Unified high-level workflow functions for training and evaluating pipelines.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import torch

from ..datasets import DialogueDataset
from ..logger import logger
from ..metrics import auroc, get_metric_by_name, recall_at_fpr, with_bootstrap
from ..pipeline import Pipeline
from ..preprocessing.pre_transforms import Pool, SelectLayer, SelectLayers
from ..processing import collect_activations
from ..processing.activations import ActivationIterator, Activations, CollectionStrategy
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


def _detect_collection_strategy_from_pipelines(
    pipelines: PipelineInput,
) -> CollectionStrategy | None:
    """Detect if all pipelines use the same sequence pooling before probe.

    Returns the pooling method if all pipelines use the same one,
    None if pipelines need dense collection (mixed methods, no pooling,
    or have post-transforms that need token-level data).

    Args:
        pipelines: Single Pipeline or mapping name → Pipeline instance

    Returns:
        "mean", "max", or "last_token" if all pipelines use the same pooling,
        None otherwise (dense collection required).
    """
    if isinstance(pipelines, Pipeline):
        pipelines_dict: Mapping[str, Pipeline] = {"_single": pipelines}
    else:
        pipelines_dict = pipelines

    pooling_methods: set[str] = set()

    for pipeline in pipelines_dict.values():
        if pipeline._post_steps:  # Post-transforms need token-level data
            return None

        method: str | None = None
        for name, transformer in pipeline._pre_steps:
            if isinstance(transformer, Pool) and transformer.dim == "sequence":
                method = transformer.method.value  # "mean", "max", or "last_token"
                break

        if method is None:
            # This pipeline needs sequences (no pre-probe pooling)
            return None

        pooling_methods.add(method)

    # All pipelines must use the same method
    if len(pooling_methods) == 1:
        return list(pooling_methods)[0]  # type: ignore[return-value]

    return None  # Mixed methods need dense collection


def _create_pipeline_without_pooling(pipeline: Pipeline) -> Pipeline:
    """Create a copy of the pipeline with sequence Pool removed.

    Used when collection_strategy already pooled the activations, so we
    skip the redundant Pool(dim="sequence") step.

    Args:
        pipeline: Original pipeline

    Returns:
        New pipeline with sequence Pool step removed
    """
    new_steps = []
    for name, step in pipeline.steps:
        if isinstance(step, Pool) and step.dim == "sequence":
            continue  # Skip - already done during collection
        new_steps.append((name, step))
    return Pipeline(new_steps)


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


def train_pipelines(
    pipelines: PipelineInput,
    activations: Activations,
    labels: torch.Tensor | list[Label],
    verbose: bool = True,
) -> None:
    """Train one or many pipelines on pre-collected activations.

    This is the core training function. Activations should be collected
    separately using :func:`collect_activations`.

    Args:
        pipelines: Single Pipeline or mapping name → Pipeline instance
        activations: Pre-collected activations (must contain all required layers)
        labels: Training labels (Tensor or list of Label enums/ints)
        verbose: Whether to print progress information

    Examples:
        >>> # Step 1: Collect activations explicitly
        >>> acts = pl.collect_activations(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     dataset=train_dataset,
        ...     layers=[16, 24],  # Explicit layer specification
        ...     mask=pl.masks.assistant(),
        ... )
        >>>
        >>> # Step 2: Train pipeline(s) on activations
        >>> pipeline = pl.Pipeline([
        ...     ("select", pl.preprocessing.SelectLayer(16)),
        ...     ("agg", pl.preprocessing.Pool(dim="sequence", method="mean")),
        ...     ("probe", pl.probes.Logistic()),
        ... ])
        >>> pl.train_pipelines(pipeline, acts, train_dataset.labels)
    """
    is_single_pipeline = isinstance(pipelines, Pipeline)
    pipelines_dict = {"_single": pipelines} if is_single_pipeline else pipelines

    if isinstance(labels, list) and labels and isinstance(labels[0], Label):
        labels_tensor = torch.tensor([label.value for label in labels])
    else:
        labels_tensor = (
            torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        )

    _train_pipelines_batch(pipelines_dict, activations, labels_tensor, verbose)


def train_pipelines_streaming(
    pipelines: PipelineInput,
    activations_iter: ActivationIterator,
    labels: torch.Tensor | list[Label],
    verbose: bool = True,
) -> None:
    """Train one or many pipelines on streaming activations (single pass).

    For large datasets that don't fit in memory. Each pipeline must support
    partial_fit() for incremental learning. Each batch is processed exactly
    once - no multi-epoch training.

    For multi-epoch training, use streaming=False (batch mode) where the
    probe's fit() method handles epochs internally.

    Args:
        pipelines: Single Pipeline or mapping name → Pipeline instance
        activations_iter: ActivationIterator yielding activation batches
        labels: All labels (indexed by batch_indices from activations)
        verbose: Whether to print progress information

    Raises:
        NotImplementedError: If any pipeline doesn't support partial_fit

    Example:
        >>> acts_iter = pl.collect_activations(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     dataset=large_dataset,
        ...     layers=[16],
        ...     mask=pl.masks.assistant(),
        ...     streaming=True,  # Returns iterator
        ... )
        >>> pl.train_pipelines_streaming(pipeline, acts_iter, large_dataset.labels)
    """
    is_single_pipeline = isinstance(pipelines, Pipeline)
    pipelines_dict = {"_single": pipelines} if is_single_pipeline else pipelines

    if verbose:
        logger.info(f"Training {len(pipelines_dict)} pipeline(s) in streaming mode (single pass)")

    for name, pipeline in pipelines_dict.items():
        if verbose and not is_single_pipeline:
            logger.info(f"Training pipeline '{name}'")

        pipeline.fit_streaming(
            activations_iter,
            labels,
            verbose=verbose,
        )


def train_from_model(
    pipelines: PipelineInput,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: DialogueDataset,
    *,
    layers: list[int],
    mask: "MaskFunction",
    batch_size: int = 32,
    streaming: bool = False,
    verbose: bool = True,
    **activation_kwargs: Any,
) -> None:
    """Convenience function: collect activations + train in one call.

    This function combines activation collection and training for convenience.
    For more control, use :func:`collect_activations` and :func:`train_pipelines`
    separately.

    **Optimization**: When all pipelines use the same sequence aggregation method
    (e.g., Pool(dim="sequence", method="mean")), activations are pooled during collection
    for ~440x memory reduction and ~2x throughput improvement.

    **Streaming vs Batch Mode**:
    - streaming=False (default): Collects all activations, then calls fit() which
      handles epochs internally (e.g., MLP trains for n_epochs=100 by default)
    - streaming=True: Processes each batch exactly once via partial_fit().
      Use for very large datasets that don't fit in memory.

    Args:
        pipelines: Single Pipeline or mapping name → Pipeline instance
        model: Language model to extract activations from
        tokenizer: Tokenizer for the model
        dataset: DialogueDataset containing dialogues and labels
        layers: Layer indices to collect (REQUIRED - no magic extraction)
        mask: Mask function for token selection (REQUIRED)
        batch_size: Batch size for activation collection
        streaming: Whether to use streaming mode (single pass with partial_fit)
        verbose: Whether to show progress
        **activation_kwargs: Additional args for collect_activations
            (e.g., hook_point, add_generation_prompt)

    Example:
        >>> # Convenience one-liner (wraps 2-step process)
        >>> pl.train_from_model(
        ...     pipeline, model, tokenizer, dataset,
        ...     layers=[16],  # Explicit layers required
        ...     mask=pl.masks.assistant(),
        ... )
        >>>
        >>> # Equivalent 2-step (more flexible):
        >>> acts = pl.collect_activations(model, tokenizer, dataset,
        ...                               layers=[16], mask=pl.masks.assistant())
        >>> pl.train_pipelines(pipeline, acts, dataset.labels)
    """
    # Detect optimal collection strategy from pipeline structure
    collection_strategy = _detect_collection_strategy_from_pipelines(pipelines)

    if verbose and collection_strategy:
        logger.info(f"Auto-detected collection strategy: pooled ({collection_strategy})")

    # 1. Collect activations (with optimized strategy if applicable)
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=layers,
        mask=mask,
        batch_size=batch_size,
        streaming=streaming,
        collection_strategy=collection_strategy,
        verbose=verbose,
        detach_activations=activation_kwargs.pop("detach_activations", True),
        **activation_kwargs,
    )

    # 2. Train pipelines on activations
    if isinstance(activations, ActivationIterator):
        train_pipelines_streaming(
            pipelines,
            activations,
            dataset.labels,
            verbose=verbose,
        )
    else:
        # If we used pooled collection, create modified pipelines without Pool(dim="sequence")
        if collection_strategy is not None:
            is_single_pipeline = isinstance(pipelines, Pipeline)
            if is_single_pipeline:
                modified_pipelines: PipelineInput = _create_pipeline_without_pooling(pipelines)
            else:
                modified_pipelines = {
                    name: _create_pipeline_without_pooling(p)
                    for name, p in pipelines.items()
                }
            train_pipelines(modified_pipelines, activations, dataset.labels, verbose)
        else:
            train_pipelines(pipelines, activations, dataset.labels, verbose)


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


def evaluate_pipelines(
    pipelines: PipelineInput,
    activations: Activations,
    labels: torch.Tensor | list[Label],
    metrics: list[Callable | str] | None = None,
    bootstrap: bool | dict[str, Any] | None = None,
) -> tuple[PredictionsOutput, MetricsOutput]:
    """Evaluate one or many pipelines on pre-collected activations.

    This is the core evaluation function. Activations should be collected
    separately using :func:`collect_activations`.

    Args:
        pipelines: Single Pipeline or mapping name → Pipeline instance
        activations: Pre-collected activations (must contain all required layers)
        labels: True labels (Tensor or list of Label enums/ints)
        metrics: List of metric functions or names (defaults to auroc + recall@1%/0.1%)
        bootstrap: None (default) disables bootstrap; True uses defaults;
            dict provides custom bootstrap kwargs

    Returns:
        Tuple of (predictions, metrics) where:
        - predictions: Tensor (single pipeline) or dict of tensors (multiple)
        - metrics: Dict (single) or dict of dicts (multiple) with metric results

    Example:
        >>> # Step 1: Collect activations
        >>> acts = pl.collect_activations(
        ...     model, tokenizer, test_dataset,
        ...     layers=[16], mask=pl.masks.assistant()
        ... )
        >>>
        >>> # Step 2: Evaluate pipeline(s)
        >>> preds, metrics = pl.evaluate_pipelines(
        ...     pipeline, acts, test_dataset.labels
        ... )
        >>> print(f"AUROC: {metrics['auroc']:.3f}")
    """
    is_single_pipeline = isinstance(pipelines, Pipeline)
    pipelines_dict = {"_single": pipelines} if is_single_pipeline else pipelines

    if isinstance(labels, list) and labels and isinstance(labels[0], Label):
        labels_tensor = torch.tensor([label.value for label in labels])
    else:
        labels_tensor = (
            torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        )

    if metrics is None:
        metrics = [
            auroc,
            functools.partial(recall_at_fpr, fpr=0.001),  # recall@0.1%
            functools.partial(recall_at_fpr, fpr=0.01),  # recall@1%
        ]
    else:
        converted_metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                converted_metrics.append(get_metric_by_name(metric))
            else:
                converted_metrics.append(metric)
        metrics = converted_metrics

    if isinstance(bootstrap, bool):
        bootstrap_kwargs: dict[str, Any] | None = {} if bootstrap else None
    else:
        bootstrap_kwargs = bootstrap

    all_predictions, all_metrics = _evaluate_pipelines_batch(
        pipelines_dict, activations, labels_tensor, metrics, bootstrap_kwargs
    )

    if is_single_pipeline:
        return all_predictions["_single"], all_metrics["_single"]
    else:
        return all_predictions, all_metrics


def evaluate_from_model(
    pipelines: PipelineInput,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: DialogueDataset,
    *,
    layers: list[int],
    mask: "MaskFunction",
    batch_size: int = 32,
    streaming: bool = False,
    metrics: list[Callable | str] | None = None,
    bootstrap: bool | dict[str, Any] | None = None,
    verbose: bool = True,
    **activation_kwargs: Any,
) -> tuple[PredictionsOutput, MetricsOutput]:
    """Convenience function: collect activations + evaluate in one call.

    This function combines activation collection and evaluation for convenience.
    For more control, use :func:`collect_activations` and :func:`evaluate_pipelines`
    separately.

    **Optimization**: When all pipelines use the same sequence aggregation method
    (e.g., Pool(dim="sequence", method="mean")), activations are pooled during collection
    for ~440x memory reduction and ~2x throughput improvement.

    Args:
        pipelines: Single Pipeline or dict mapping names to Pipelines
        model: Language model to extract activations from
        tokenizer: Tokenizer for the model
        dataset: DialogueDataset containing dialogues and labels
        layers: Layer indices to collect (REQUIRED - no magic extraction)
        mask: Mask function for token selection (REQUIRED)
        batch_size: Batch size for activation collection
        streaming: Whether to use streaming mode
        metrics: List of metric functions or names
        bootstrap: Bootstrap configuration
        verbose: Whether to show progress
        **activation_kwargs: Additional args for collect_activations

    Returns:
        Tuple of (predictions, metrics)

    Example:
        >>> # Convenience one-liner
        >>> preds, metrics = pl.evaluate_from_model(
        ...     pipeline, model, tokenizer, test_dataset,
        ...     layers=[16], mask=pl.masks.assistant()
        ... )
    """
    # Detect optimal collection strategy from pipeline structure
    collection_strategy = _detect_collection_strategy_from_pipelines(pipelines)

    if verbose and collection_strategy:
        logger.info(f"Auto-detected collection strategy: pooled ({collection_strategy})")

    # 1. Collect activations (with optimized strategy if applicable)
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=layers,
        mask=mask,
        batch_size=batch_size,
        streaming=streaming,
        collection_strategy=collection_strategy,
        verbose=verbose,
        detach_activations=activation_kwargs.pop("detach_activations", True),
        **activation_kwargs,
    )

    # 2. Evaluate pipelines on activations
    if isinstance(activations, ActivationIterator):
        raise NotImplementedError(
            "Streaming evaluation not yet implemented. "
            "Use streaming=False or implement evaluate_pipelines_streaming()."
        )
    else:
        # If we used pooled collection, create modified pipelines without Pool(dim="sequence")
        if collection_strategy is not None:
            is_single_pipeline = isinstance(pipelines, Pipeline)
            if is_single_pipeline:
                modified_pipelines: PipelineInput = _create_pipeline_without_pooling(pipelines)
            else:
                modified_pipelines = {
                    name: _create_pipeline_without_pooling(p)
                    for name, p in pipelines.items()
                }
            return evaluate_pipelines(modified_pipelines, activations, dataset.labels, metrics, bootstrap)
        else:
            return evaluate_pipelines(pipelines, activations, dataset.labels, metrics, bootstrap)


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

        # Convert to numpy for metrics (ensure float32 for numpy compatibility)
        y_true = (
            labels.detach().cpu().float().numpy()
            if isinstance(labels, torch.Tensor)
            else labels
        )
        y_pred = (
            preds.detach().cpu().float().numpy() if isinstance(preds, torch.Tensor) else preds
        )

        for metric_fn in metrics:
            metric_name = _metric_display_name(metric_fn)
            metric_callable = metric_fn
            if bootstrap_kwargs is not None and not getattr(
                metric_fn, "_probelab_bootstrap", False
            ):
                metric_callable = with_bootstrap(**bootstrap_kwargs)(metric_fn)

            # Compute metric (optionally bootstrapped)
            result = metric_callable(y_true, y_pred)
            pipeline_metrics[metric_name] = result

        all_metrics[name] = pipeline_metrics

    return all_predictions, all_metrics
