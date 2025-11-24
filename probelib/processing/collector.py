"""Reusable activation collector for reducing boilerplate.

This module provides the ActivationCollector class, which encapsulates
model, tokenizer, and collection settings for efficient reuse across
multiple datasets.
"""

from typing import TYPE_CHECKING, Any, Literal

import torch

from .activations import ActivationIterator, Activations, collect_activations

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..datasets import DialogueDataset
    from ..masks import MaskFunction


class ActivationCollector:
    """Reusable activation collector with sensible defaults.

    This class encapsulates model, tokenizer, and common collection parameters,
    allowing efficient reuse across multiple datasets without repeating
    configuration. It provides convenience methods for different collection
    modes (dense, pooled, streaming) while maintaining compatibility with
    the underlying ``collect_activations()`` function.

    Key Benefits:
        - **Eliminate boilerplate**: Configure once, use many times
        - **Consistent settings**: Ensure same parameters across experiments
        - **Readable code**: Clear intent with named methods
        - **Type-safe**: Full type hints for IDE support

    Args:
        model: Language model whose activations are collected.
        tokenizer: Tokenizer aligned with the model.
        layers: Layer index or indices to record. Can be single int or list.
        batch_size: Number of sequences per activation batch. Default: 32.
        hook_point: Where to extract activations:
            - ``"post_block"`` (default): After transformer block + layernorm
            - ``"pre_layernorm"``: Before layer normalization
        detach_activations: Whether to detach from computation graph. Default: True.
            Set to False only if you need gradients through activations.
        device: Target device for model. If None, auto-detects (cuda if available).
        verbose: Whether to show progress bars during collection. Default: True.

    Attributes:
        model: The language model.
        tokenizer: The tokenizer.
        layers: List of layer indices to collect.
        batch_size: Batch size for collection.
        hook_point: Hook point for activation extraction.
        detach_activations: Whether activations are detached.
        device: Device for computation.
        verbose: Verbosity flag.

    Examples:
        Basic usage with single dataset:

        >>> import probelib as pl
        >>> collector = pl.ActivationCollector(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     layers=[16, 20, 24],
        ...     batch_size=32
        ... )
        >>> acts = collector.collect(dataset)

        Reuse across multiple datasets:

        >>> # Configure once
        >>> collector = pl.ActivationCollector(model, tokenizer, layers=[16])
        >>>
        >>> # Reuse many times
        >>> train_acts = collector.collect(train_dataset)
        >>> test_acts = collector.collect(test_dataset)
        >>> val_acts = collector.collect(val_dataset)

        Explicit collection modes:

        >>> # Dense collection (full sequences)
        >>> dense = collector.collect_dense(dataset)
        >>> assert dense.has_axis(pl.processing.Axis.SEQ)
        >>>
        >>> # Pooled collection (memory efficient)
        >>> pooled = collector.collect_pooled(dataset, method="mean")
        >>> assert not pooled.has_axis(pl.processing.Axis.SEQ)
        >>>
        >>> # Streaming collection (very large datasets)
        >>> streaming = collector.collect_streaming(large_dataset)
        >>> assert isinstance(streaming, pl.processing.ActivationIterator)

        Compare with function-based approach:

        >>> # Old: function-based (verbose, error-prone)
        >>> acts1 = pl.collect_activations(
        ...     model, tokenizer, dataset1, layers=[16], batch_size=8, verbose=True
        ... )
        >>> acts2 = pl.collect_activations(
        ...     model, tokenizer, dataset2, layers=[16], batch_size=8, verbose=True
        ... )
        >>>
        >>> # New: class-based (concise, reusable)
        >>> collector = pl.ActivationCollector(model, tokenizer, layers=[16], batch_size=8)
        >>> acts1 = collector.collect(dataset1)
        >>> acts2 = collector.collect(dataset2)
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
        *,
        layers: int | list[int],
        batch_size: int = 32,
        hook_point: Literal["pre_layernorm", "post_block"] = "post_block",
        detach_activations: bool = True,
        device: str | None = None,
        verbose: bool = True,
    ):
        """Initialize the activation collector.

        Args:
            model: Language model whose activations are collected.
            tokenizer: Tokenizer aligned with the model.
            layers: Layer index or indices to record.
            batch_size: Number of sequences per batch. Default: 32.
            hook_point: Activation extraction point. Default: "post_block".
            detach_activations: Whether to detach from graph. Default: True.
            device: Target device. If None, auto-detects.
            verbose: Show progress bars. Default: True.
        """
        self.model = model
        self.tokenizer = tokenizer

        # Normalize layers to list
        self.layers = [layers] if isinstance(layers, int) else list(layers)

        self.batch_size = batch_size
        self.hook_point = hook_point
        self.detach_activations = detach_activations
        self.verbose = verbose

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def collect(
        self,
        dataset: "DialogueDataset",
        *,
        mask: "MaskFunction | None" = None,
        streaming: bool | Literal["auto"] = "auto",
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Activations | ActivationIterator:
        """Collect activations from a dataset.

        This is the main collection method that delegates to ``collect_activations()``.
        It provides sensible defaults and auto-detection while allowing full
        customization via parameters.

        Note: Aggregation (pooling) should be done using Pipeline preprocessing
        transformers after collection. Use ``collect_pooled()`` for a convenience
        method that pools immediately.

        Args:
            dataset: Dataset containing dialogues.
            mask: Optional mask for token selection. If None, uses
                ``dataset.default_mask`` if dataset is a DialogueDataset.
            streaming: Collection mode:
                - ``"auto"`` (default): Automatically choose based on dataset size
                  (streaming if len(dataset) > 10,000)
                - ``True``: Force streaming mode (returns ActivationIterator)
                - ``False``: Force batch mode (returns Activations)
            add_generation_prompt: Whether to append generation tokens.
            **kwargs: Additional arguments passed to ``collect_activations()``.

        Returns:
            - ``Activations``: Dense collection with full sequences [layers, batch, seq, hidden]
            - ``ActivationIterator``: For streaming collection

        Examples:
            Default behavior (auto-streaming):

            >>> acts = collector.collect(dataset)

            Force batch collection:

            >>> acts = collector.collect(dataset, streaming=False)

            Custom mask override:

            >>> acts = collector.collect(
            ...     dataset,
            ...     mask=pl.masks.AndMask(
            ...         pl.masks.assistant(),
            ...         pl.masks.last_n_tokens(10)
            ...     )
            ... )
        """
        # Import here to avoid circular imports
        from ..datasets import DialogueDataset

        # Auto-detect streaming based on dataset size
        if streaming == "auto":
            # Use streaming for datasets larger than 10k samples
            streaming = len(dataset) > 10_000

        # Use dataset's default mask if not provided
        if mask is None and isinstance(dataset, DialogueDataset):
            mask = dataset.default_mask

        # Delegate to collect_activations function (always dense collection)
        return collect_activations(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            layers=self.layers,
            batch_size=self.batch_size,
            mask=mask,
            streaming=streaming,
            verbose=self.verbose,
            hook_point=self.hook_point,
            detach_activations=self.detach_activations,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def collect_dense(
        self,
        dataset: "DialogueDataset",
        mask: "MaskFunction | None" = None,
        **kwargs: Any,
    ) -> Activations:
        """Collect activations with full sequences (dense mode).

        This method explicitly disables streaming to ensure you get full sequence
        activations with shape ``[layers, batch, seq, hidden]``. Use this when
        you need access to individual token representations.

        Args:
            dataset: Dataset containing dialogues.
            mask: Optional mask for token selection.
            **kwargs: Additional arguments passed to ``collect_activations()``.

        Returns:
            Activations object with sequence dimension present.

        Examples:
            >>> acts = collector.collect_dense(dataset)
            >>> print(acts.axes)  # (LAYER, BATCH, SEQ, HIDDEN)
            >>> print(acts.shape)  # [3, 1000, 512, 4096]
        """
        result = self.collect(
            dataset, mask=mask, streaming=False, **kwargs
        )
        assert isinstance(result, Activations), "Expected Activations, got Iterator"
        return result

    def collect_pooled(
        self,
        dataset: "DialogueDataset",
        mask: "MaskFunction | None" = None,
        method: Literal["mean", "max", "last_token"] = "mean",
        **kwargs: Any,
    ) -> Activations:
        """Collect activations and pool over sequences (memory efficient).

        This method collects dense activations then immediately pools over the
        sequence dimension, resulting in activations with shape
        ``[layers, batch, hidden]`` (no sequence dimension). This provides
        ~440x memory reduction compared to dense collection.

        Use this when you only need sequence-level representations and want to
        minimize memory usage.

        Note: For more flexibility (e.g., pooling only certain layers), use
        Pipeline preprocessing transformers instead.

        Args:
            dataset: Dataset containing dialogues.
            mask: Optional mask for token selection.
            method: Pooling method:
                - ``"mean"`` (default): Average over sequence
                - ``"max"``: Maximum over sequence
                - ``"last_token"``: Use only last token
            **kwargs: Additional arguments passed to ``collect_activations()``.

        Returns:
            Activations object without sequence dimension.

        Examples:
            >>> acts = collector.collect_pooled(dataset, method="mean")
            >>> print(acts.axes)  # (LAYER, BATCH, HIDDEN)
            >>> print(acts.shape)  # [3, 1000, 4096]
            >>>
            >>> # 440x memory reduction!
            >>> # Dense: 240 GB → Pooled: 540 MB
        """
        # Collect dense activations
        result = self.collect(
            dataset,
            mask=mask,
            streaming=False,
            **kwargs,
        )
        assert isinstance(result, Activations), "Expected Activations, got Iterator"

        # Pool over sequence dimension
        return result.pool(dim="sequence", method=method, use_detection_mask=True)

    def collect_streaming(
        self,
        dataset: "DialogueDataset",
        mask: "MaskFunction | None" = None,
        **kwargs: Any,
    ) -> ActivationIterator:
        """Collect activations in streaming mode (for large datasets).

        This method forces streaming mode, returning an iterator that yields
        activation batches lazily. Use this for very large datasets that don't
        fit in memory, or when you want to make multiple passes over the data.

        The returned iterator can be used in for loops and supports multiple
        iterations (creates fresh generator each time).

        Args:
            dataset: Dataset containing dialogues.
            mask: Optional mask for token selection.
            **kwargs: Additional arguments passed to ``collect_activations()``.

        Returns:
            ActivationIterator that yields Activations batches.

        Examples:
            >>> acts_iter = collector.collect_streaming(large_dataset)
            >>>
            >>> # First pass: compute statistics
            >>> for batch in acts_iter:
            ...     update_statistics(batch)
            >>>
            >>> # Second pass: train probe incrementally
            >>> for batch in acts_iter:  # Creates fresh iterator
            ...     probe.partial_fit(batch, labels[batch.batch_indices])
        """
        result = self.collect(dataset, mask=mask, streaming=True, **kwargs)
        assert isinstance(
            result, ActivationIterator
        ), "Expected Iterator, got Activations"
        return result

    def info(self) -> dict[str, Any]:
        """Return collector configuration as dictionary.

        Returns:
            Dictionary with collector settings.

        Examples:
            >>> collector = pl.ActivationCollector(model, tokenizer, layers=[16, 20])
            >>> print(collector.info())
            {
                'layers': [16, 20],
                'batch_size': 32,
                'hook_point': 'post_block',
                'detach_activations': True,
                'device': 'cuda',
                'verbose': True
            }
        """
        return {
            "layers": self.layers,
            "batch_size": self.batch_size,
            "hook_point": self.hook_point,
            "detach_activations": self.detach_activations,
            "device": self.device,
            "verbose": self.verbose,
        }

    def __repr__(self) -> str:
        """String representation of the collector."""
        layers_str = (
            f"[{', '.join(map(str, self.layers))}]"
            if len(self.layers) > 1
            else str(self.layers[0])
        )
        return (
            f"ActivationCollector(layers={layers_str}, "
            f"batch_size={self.batch_size}, "
            f"device={self.device})"
        )
