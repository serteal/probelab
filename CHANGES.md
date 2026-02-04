# Planned Changes: sklearn-style probelab with `pre`/`post` transforms

This document tracks planned changes to probelab's architecture. Do not implement until finalized.

## Overview

Adopt sklearn-style patterns:
- Centralized validation via `check_activations()` / `check_scores()`
- Type-based transform classification (`pre.*` for Activations→Activations, `post.*` for Scores→Scores)
- Duck typing in Pipeline (check methods, not isinstance)
- Fail fast with helpful error messages

---

## 1. New Validation Module

**Create `probelab/utils/validation.py`** with centralized validation functions:

```python
def check_activations(
    X: Activations,
    *,
    require_layer: bool = False,
    forbid_layer: bool = False,
    require_seq: bool = False,
    forbid_seq: bool = False,
    ensure_finite: bool = True,
    ensure_non_empty: bool = True,
    estimator_name: str = "",
) -> Activations:
    """Validate Activations object."""
    ...

def check_scores(
    X: Scores,
    *,
    require_seq: bool = False,
    forbid_seq: bool = False,
    ensure_finite: bool = True,
    estimator_name: str = "",
) -> Scores:
    """Validate Scores object."""
    ...
```

All probes and transforms call these instead of inline validation. Provides consistent error messages and single place for new validation logic.

---

## 2. Restructure Transforms Module

**Rename and reorganize `probelab/transforms/`:**

| Old | New |
|-----|-----|
| `transforms/base.py` (PreTransformer) | `transforms/base.py` (ActivationTransform, ScoreTransform) |
| `transforms/pre_transforms.py` | `transforms/pre.py` |
| (none) | `transforms/post.py` |

**New base classes in `transforms/base.py`:**

```python
class ActivationTransform(ABC):
    """Transform: Activations → Activations."""

    def fit(self, X: Activations, y=None) -> "ActivationTransform":
        return self

    @abstractmethod
    def transform(self, X: Activations) -> Activations:
        pass

    def fit_transform(self, X: Activations, y=None) -> Activations:
        return self.fit(X, y).transform(X)


class ScoreTransform(ABC):
    """Transform: Scores → Scores."""

    def fit(self, X: Scores, y=None) -> "ScoreTransform":
        return self

    @abstractmethod
    def transform(self, X: Scores) -> Scores:
        pass

    def fit_transform(self, X: Scores, y=None) -> Scores:
        return self.fit(X, y).transform(X)
```

**Contents of `transforms/pre.py`** (all inherit from ActivationTransform):
- `SelectLayer` - select single layer, removes LAYER axis
- `SelectLayers` - select multiple layers, keeps LAYER axis
- `Pool` - pool over "layer" or "sequence" dimension
- `Normalize` - normalize features

**Contents of `transforms/post.py`** (all inherit from ScoreTransform):
- `Pool` - pool scores over sequence dimension

**Exports in `transforms/__init__.py`:**

```python
from . import pre
from . import post
from .base import ActivationTransform, ScoreTransform
```

---

## 3. Update Pipeline Validation

**Change `probelab/pipeline.py`** to use:

- **Duck typing** for probe detection: check for `predict_proba` method instead of `isinstance(step, BaseProbe)`
- **Type-based validation** for transforms: steps before probe must be `ActivationTransform`, steps after must be `ScoreTransform`
- **Better error messages** that tell users which module to use (`pre.*` or `post.*`)

```python
def _validate_steps(self):
    # Find probe by method existence (duck typing)
    probe_indices = [
        i for i, (name, step) in enumerate(self.steps)
        if hasattr(step, 'predict_proba') and callable(step.predict_proba)
    ]

    # Validate pre-transforms (before probe)
    for i, (name, step) in enumerate(self.steps[:probe_idx]):
        if not isinstance(step, ActivationTransform):
            raise TypeError(
                f"Step '{name}' (position {i}) must be an ActivationTransform (pre.*), "
                f"got {type(step).__name__}"
            )

    # Validate post-transforms (after probe)
    for i, (name, step) in enumerate(self.steps[probe_idx + 1:], start=probe_idx + 1):
        if not isinstance(step, ScoreTransform):
            raise TypeError(
                f"Step '{name}' (position {i}) must be a ScoreTransform (post.*), "
                f"got {type(step).__name__}"
            )
```

---

## 4. Update Probes

**Modify probes in `probelab/probes/`** to use centralized validation:

```python
# Before (scattered validation)
def fit(self, X: Activations, y):
    if X.has_axis(Axis.LAYER):
        raise ValueError("Logistic probe expects single layer activations...")

# After (centralized)
def fit(self, X: Activations, y):
    X = check_activations(X, forbid_layer=True, estimator_name="Logistic")
```

Files to modify:
- `logistic.py`
- `mlp.py`
- `attention.py`

---

## 5. Simplify Probe/Pipeline API: Remove `predict_proba()`, Keep Only `predict()`

**Change:** Remove `predict_proba()` method. `predict()` returns probabilities `[batch, 2]`.

**Rationale:** Simpler API - one method instead of two. Users can derive class labels easily.

**Before:**
```python
# Two methods
probs = pipeline.predict_proba(X)  # [batch, 2]
preds = pipeline.predict(X)         # [batch] class labels
```

**After:**
```python
# One method
probs = pipeline.predict(X)         # [batch, 2] probabilities

# Users derive class labels if needed
preds = probs.argmax(dim=1)         # [batch] class labels
preds = (probs[:, 1] > 0.5).long()  # [batch] with custom threshold
```

**Changes to BaseProbe:**
```python
class BaseProbe(ABC):
    @abstractmethod
    def fit(self, X: Activations, y) -> "BaseProbe":
        pass

    @abstractmethod
    def predict(self, X: Activations) -> Scores:  # Renamed from predict_proba
        """Predict class probabilities.

        Returns:
            Scores with shape [batch, 2] or [batch, seq, 2]
        """
        pass

    # REMOVED: predict_proba() - use predict() instead
```

**Changes to Pipeline:**
```python
class Pipeline:
    def predict(self, X: Activations) -> torch.Tensor:
        """Predict class probabilities.

        Returns:
            Probabilities [batch, 2]
        """
        # ... apply pre-transforms ...
        scores = self._probe.predict(X_transformed)
        # ... apply post-transforms ...
        return scores.scores

    # REMOVED: predict_proba() - use predict() instead

    def score(self, X: Activations, y) -> float:
        """Compute accuracy."""
        probs = self.predict(X)
        preds = probs.argmax(dim=1)
        # ...
```

**Changes to Pipeline validation (duck typing):**
```python
# Before: check for predict_proba
probe_indices = [
    i for i, (name, step) in enumerate(self.steps)
    if hasattr(step, 'predict_proba')
]

# After: check for predict
probe_indices = [
    i for i, (name, step) in enumerate(self.steps)
    if hasattr(step, 'predict') and hasattr(step, 'fit')
]
```

**Migration:**
```python
# Old
probs = pipeline.predict_proba(X)
preds = pipeline.predict(X)

# New
probs = pipeline.predict(X)
preds = probs.argmax(dim=1)
```

---

## 6. Update Public API Exports

**Modify `probelab/__init__.py`:**

- Export `check_activations`, `check_scores` from utils
- Update transform exports to use new structure

Old:
```python
from probelab.transforms import SelectLayer, Pool, Normalize
```

New:
```python
from probelab.transforms import pre, post
# Use as: pre.SelectLayer, pre.Pool, post.Pool
```

---

## 6. Update Tests

- **Rename** `tests/transforms/test_pre_transforms.py` → `tests/transforms/test_pre.py`
- **Create** `tests/transforms/test_post.py`
- **Create** `tests/utils/test_validation.py`
- **Update** `tests/test_pipeline.py` to test type-based validation
- **Update** error message assertions throughout

---

## 7. Update CLAUDE.md Documentation

Update to reflect:
- New import patterns (`from probelab.transforms import pre, post`)
- New validation functions
- Pipeline structure explanation (ActivationTransform* → Probe → ScoreTransform*)

---

## 8. Update Activations Class (Keep Axes, Move Validation Out)

**Philosophy:** Activations keeps axis tracking for introspection, but removes validation from its methods. Validation is the caller's responsibility (via `check_activations()`).

**Modify `probelab/processing/activations.py`:**

```python
# Before - Activations validates internally
class Activations:
    axes: tuple[Axis, ...]  # Keep this

    def select(self, layer=None, layers=None):
        if layer is not None and layers is not None:
            raise ValueError("Cannot specify both layer and layers")
        if not self.has_axis(Axis.LAYER):
            raise ValueError("Activations don't have LAYER axis")  # REMOVE THIS
        ...

    def pool(self, dim: str, method: str):
        if dim == "sequence" and not self.has_axis(Axis.SEQ):
            raise ValueError("No SEQ axis to pool over")  # REMOVE THIS
        ...

# After - Activations trusts caller, no validation
class Activations:
    axes: tuple[Axis, ...]  # Keep for introspection

    def select(self, layer=None, layers=None):
        if layer is not None and layers is not None:
            raise ValueError("Cannot specify both layer and layers")
        # No axis validation - caller's responsibility
        ...

    def pool(self, dim: str, method: str):
        # No axis validation - caller's responsibility
        ...
```

**Why keep axes?**
- `check_activations()` needs to inspect `X.has_axis(Axis.LAYER)`
- Better error messages: "missing LAYER axis" vs "expected 4D"
- Self-documenting: `X.axes` tells you what dimensions mean
- Properties like `n_layers`, `seq_len` depend on knowing which axes exist

**What validation to remove from Activations:**

| Method | Remove |
|--------|--------|
| `select(layer=...)` | Axis.LAYER existence check |
| `select(layers=...)` | Axis.LAYER existence check |
| `pool(dim="layer")` | Axis.LAYER existence check |
| `pool(dim="sequence")` | Axis.SEQ existence check |

**Keep these validations in Activations** (not axis-related):
- `select()`: "Cannot specify both layer and layers"
- `select(layer=X)`: "Layer X not in layer_indices"
- Basic argument validation

**Same pattern for Scores:**

```python
# Scores.pool() - remove axis validation
def pool(self, dim: str, method: str):
    # No axis validation - caller's responsibility
    ...
```

---

## What Stays the Same

- `Activations.axes` attribute and `has_axis()` method (kept for introspection)
- `Scores` class structure
- `collect_activations()` function
- Dataset classes and registry
- Metrics module
- Masks module

---

## Migration Guide for Users

### Transform imports

Old:
```python
from probelab.transforms import SelectLayer, Pool
```

New:
```python
from probelab.transforms import pre, post
```

### Pipeline construction

Old:
```python
from probelab import Pipeline
from probelab.transforms import SelectLayer, Pool
from probelab.probes import Logistic

Pipeline([
    ("select", SelectLayer(16)),
    ("pool", Pool(dim="sequence", method="mean")),
    ("probe", Logistic()),
])
```

New:
```python
from probelab import Pipeline
from probelab.transforms import pre, post
from probelab.probes import Logistic

Pipeline([
    ("select", pre.SelectLayer(16)),
    ("pool", pre.Pool(dim="sequence", method="mean")),
    ("probe", Logistic()),
])
```

Token-level with aggregation:
```python
Pipeline([
    ("select", pre.SelectLayer(16)),
    ("probe", Logistic()),
    ("agg", post.Pool(method="mean")),
])
```

### Prediction

Old:
```python
probs = pipeline.predict_proba(X)  # [batch, 2]
preds = pipeline.predict(X)         # [batch] class labels
```

New:
```python
probs = pipeline.predict(X)         # [batch, 2] probabilities
preds = probs.argmax(dim=1)         # [batch] class labels (if needed)
```

---

## Files Changed Summary

| Action | File |
|--------|------|
| Create | `probelab/utils/__init__.py` |
| Create | `probelab/utils/validation.py` |
| Create | `probelab/transforms/post.py` |
| Rename | `probelab/transforms/pre_transforms.py` → `probelab/transforms/pre.py` |
| Modify | `probelab/transforms/base.py` |
| Modify | `probelab/transforms/__init__.py` |
| Modify | `probelab/processing/activations.py` (remove axis validation from methods) |
| Modify | `probelab/processing/scores.py` (remove axis validation from methods) |
| Modify | `probelab/pipeline.py` |
| Modify | `probelab/probes/base.py` (rename predict_proba → predict) |
| Modify | `probelab/probes/logistic.py` |
| Modify | `probelab/probes/mlp.py` |
| Modify | `probelab/probes/attention.py` |
| Modify | `probelab/__init__.py` |
| Modify | `CLAUDE.md` |
| Create | `tests/utils/test_validation.py` |
| Create | `tests/transforms/test_post.py` |
| Rename | `tests/transforms/test_pre_transforms.py` → `tests/transforms/test_pre.py` |
| Modify | `tests/test_pipeline.py` |
| Modify | `tests/processing/test_activations.py` (update for no-validation behavior) |

---

## Open Questions

1. Should we keep backwards-compatible aliases for old import paths?
2. Should `check_activations` / `check_scores` be public API or internal utils?
3. Any additional post-transforms to add (Threshold, Calibrate)?
