"""Centralized validation functions for probelab.

Provides sklearn-style validation functions for Activations objects.
These are internal utilities, not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..processing.activations import Activations, Axis


def check_activations(
    X: "Activations",
    *,
    require_layer: bool = False,
    forbid_layer: bool = False,
    require_seq: bool = False,
    forbid_seq: bool = False,
    ensure_finite: bool = True,
    ensure_non_empty: bool = True,
    estimator_name: str = "",
) -> "Activations":
    """Validate an Activations object.

    This is the centralized validation function for activation data. All probes
    and transforms should use this instead of inline validation.

    Args:
        X: The Activations object to validate.
        require_layer: If True, raise if LAYER axis is missing.
        forbid_layer: If True, raise if LAYER axis is present.
        require_seq: If True, raise if SEQ axis is missing.
        forbid_seq: If True, raise if SEQ axis is present.
        ensure_finite: If True, check for NaN/Inf values.
        ensure_non_empty: If True, check that tensor is not empty.
        estimator_name: Name of the estimator for error messages.

    Returns:
        The validated Activations object (unchanged).

    Raises:
        TypeError: If X is not an Activations object.
        ValueError: If validation fails.

    Example:
        >>> X = check_activations(X, forbid_layer=True, estimator_name="Logistic")
    """
    # Import here to avoid circular imports
    from ..processing.activations import Activations, Axis

    prefix = f"{estimator_name}: " if estimator_name else ""

    # Type check
    if not isinstance(X, Activations):
        raise TypeError(f"{prefix}Expected Activations, got {type(X).__name__}")

    # Axis validation
    if require_layer and forbid_layer:
        raise ValueError("Cannot both require and forbid LAYER axis")
    if require_seq and forbid_seq:
        raise ValueError("Cannot both require and forbid SEQ axis")

    if require_layer and not X.has_axis(Axis.LAYER):
        raise ValueError(
            f"{prefix}Expected activations with LAYER axis, but it is missing.\n"
            f"Current axes: {[ax.name for ax in X.axes]}\n"
            f"Hint: Make sure you haven't removed the LAYER axis with select(layer=...)."
        )

    if forbid_layer and X.has_axis(Axis.LAYER):
        raise ValueError(
            f"{prefix}Expected single-layer activations, but found LAYER axis "
            f"with {X.n_layers} layers.\n"
            f"Available layers: {X.layer_indices}\n"
            f"Hint: Use pre.SelectLayer(layer_idx) in your pipeline to select a single layer."
        )

    if require_seq and not X.has_axis(Axis.SEQ):
        raise ValueError(
            f"{prefix}Expected activations with SEQ axis, but it is missing.\n"
            f"Current axes: {[ax.name for ax in X.axes]}\n"
            f"Hint: The SEQ axis may have been removed by pool(dim='sequence'). "
            f"This estimator requires token-level activations."
        )

    if forbid_seq and X.has_axis(Axis.SEQ):
        raise ValueError(
            f"{prefix}Expected pooled activations without SEQ axis, but SEQ axis is present.\n"
            f"Current axes: {[ax.name for ax in X.axes]}\n"
            f"Hint: Use pre.Pool(dim='sequence', method='mean') to aggregate tokens."
        )

    # Non-empty check
    if ensure_non_empty and X.activations.numel() == 0:
        raise ValueError(f"{prefix}Received empty activations tensor")

    # Finite check
    if ensure_finite:
        if not torch.isfinite(X.activations).all():
            n_nan = torch.isnan(X.activations).sum().item()
            n_inf = torch.isinf(X.activations).sum().item()
            raise ValueError(
                f"{prefix}Activations contain non-finite values: "
                f"{n_nan} NaN, {n_inf} Inf"
            )

    return X
