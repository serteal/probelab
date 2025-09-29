"""Specification dataclasses for benchmarks."""

from dataclasses import dataclass, field
from typing import Callable

from probelib.datasets.base import DialogueDataset
from probelib.masks.base import MaskFunction


@dataclass
class EvalSpec:
    """
    Specification for evaluating on a single dataset.

    Args:
        dataset: Dataset to evaluate on
        model: Model name/path (overrides benchmark default if specified)
        split: Dataset split to use (default: "test")
        mask: Mask function to apply during tokenization
        metrics: List of metric names or callables to compute
        name: Custom name for this evaluation in results table

    Example:
        >>> spec = EvalSpec(
        ...     dataset=pl.datasets.AIAuditDataset(split="test"),
        ...     mask=pl.masks.assistant(),
        ...     metrics=["auroc", "accuracy", "recall@5"],
        ...     name="AI Audit Test"
        ... )
    """

    dataset: DialogueDataset
    model: str | None = None
    split: str = "test"
    mask: MaskFunction | None = None
    metrics: list[str | Callable] = field(default_factory=lambda: ["auroc", "accuracy"])
    name: str | None = None

    def __post_init__(self):
        """Generate default name from dataset if not provided."""
        if self.name is None:
            dataset_name = self.dataset.__class__.__name__.replace("Dataset", "")
            self.name = f"{dataset_name} ({self.split})"


@dataclass
class TrainSpec:
    """
    Specification for training before evaluation.

    Args:
        datasets: List of datasets to train on
        model: Model name/path (overrides benchmark default if specified)
        mask: Mask function to apply during tokenization
        probe_config: Keyword arguments passed to probe constructor
        train_kwargs: Keyword arguments passed to train_probes()

    Example:
        >>> spec = TrainSpec(
        ...     datasets=[
        ...         pl.datasets.AIAuditDataset(split="train"),
        ...         pl.datasets.AILiarDataset(split="train")
        ...     ],
        ...     mask=pl.masks.assistant(),
        ...     probe_config={"layer": 16, "sequence_aggregation": "mean"},
        ...     train_kwargs={"batch_size": 16}
        ... )
    """

    datasets: list[DialogueDataset]
    model: str | None = None
    mask: MaskFunction | None = None
    probe_config: dict = field(default_factory=dict)
    train_kwargs: dict = field(default_factory=dict)