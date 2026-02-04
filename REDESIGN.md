# Probelab Redesign: Deep Analysis

## What is Probelab?

At its core, probelab is about:
1. Taking dialogues
2. Running them through an LLM to extract internal representations
3. Training classifiers on those representations
4. Evaluating those classifiers

## The Data Flow

```
Dialogues → Tokens → Hidden States → Features → Predictions → Metrics
         (tokenize) (collect)      (transform)  (classify)   (evaluate)
```

Each arrow is a well-defined transformation with input/output types.

## The Core Tension

The tricky part is the "Features → Predictions" step, where multiple valid paths exist:

```
Path A: Token-level training, then aggregate scores
[L,B,S,H] → select → [B,S,H] → classify → [B,S,2] → pool → [B,2]

Path B: Aggregate features, then train
[L,B,S,H] → select → [B,S,H] → pool → [B,H] → classify → [B,2]

Path C: Attention-based (requires sequence)
[L,B,S,H] → select → [B,S,H] → attention_classify → [B,2]
```

All are valid. The polymorphism is **real and useful**.

## Current Design Analysis

### What Works Well
- Activations/Scores separation is meaningful (features vs probabilities)
- Pipeline composition is explicit
- Axis tracking prevents shape bugs
- Masks are composable and clean

### What's Unclear
- Probe polymorphism is implicit (Logistic handles both [B,H] and [B,S,H])
- No explicit dimension contracts
- Hard to know at a glance what dimensions an op requires/produces

---

## Proposed Redesign

### Principle 1: Explicit Dimension Contracts

Every operation declares what it does to dimensions:

```python
class Op:
    """Base for all tensor operations."""

    # Dimension contract
    requires: tuple[Axis, ...]  # Must have these axes
    removes: tuple[Axis, ...]   # Removes these axes
    adds: tuple[Axis, ...]      # Adds these axes
    broadcasts: bool = False    # If True, works on any extra dims
```

Examples:

```python
class SelectLayer(Op):
    """Remove LAYER axis by selecting one layer."""
    requires = (LAYER,)
    removes = (LAYER,)
    adds = ()
    broadcasts = True  # Works regardless of other dims

class PoolSequence(Op):
    """Remove SEQ axis by pooling."""
    requires = (SEQ,)
    removes = (SEQ,)
    adds = ()
    broadcasts = True

class LinearClassify(Op):
    """Replace HIDDEN with CLASS. Broadcasts over all other dims."""
    requires = (HIDDEN,)
    removes = (HIDDEN,)
    adds = (CLASS,)
    broadcasts = True  # [B,H]→[B,2] or [B,S,H]→[B,S,2]

class AttentionClassify(Op):
    """Reduce SEQ via attention, then classify."""
    requires = (BATCH, SEQ, HIDDEN)
    removes = (SEQ, HIDDEN)
    adds = (CLASS,)
    broadcasts = False  # Requires exactly these dims
```

### Principle 2: Unified Op Interface

Both stateless transforms and stateful classifiers are "Ops":

```python
# Stateless op (no learnable parameters)
class SelectLayer(Op):
    def __init__(self, layer: int):
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        return x.index_select(dim=LAYER, index=self.layer)

# Stateful op (has learnable parameters)
class LinearClassify(Op):
    def __init__(self, hidden_dim: int, device: str = "cuda"):
        self.weight = torch.nn.Parameter(...)
        self.bias = torch.nn.Parameter(...)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., hidden] → [..., 2]
        return torch.sigmoid(x @ self.weight + self.bias)

    def fit(self, x: Tensor, y: Tensor) -> "LinearClassify":
        # Train the parameters
        ...
        return self
```

The only difference: stateful ops have `fit()`.

### Principle 3: Pipeline Validates Dimension Flow

```python
class Pipeline:
    def __init__(self, ops: list[Op]):
        self.ops = ops
        self._validate()

    def _validate(self):
        """Check that ops are dimensionally compatible."""
        # Track which axes are present
        axes = {LAYER, BATCH, SEQ, HIDDEN}  # Start with full activation

        for op in self.ops:
            # Check requires
            for axis in op.requires:
                if axis not in axes:
                    raise ValueError(f"{op} requires {axis} but only have {axes}")

            # Apply removes/adds
            axes -= set(op.removes)
            axes |= set(op.adds)

        # Final result should have BATCH and CLASS
        if CLASS not in axes:
            raise ValueError("Pipeline must produce CLASS axis")
```

Now invalid pipelines fail at construction:

```python
# FAILS: AttentionClassify requires SEQ, but PoolSequence removed it
Pipeline([
    SelectLayer(16),
    PoolSequence("mean"),
    AttentionClassify(),  # Error: requires SEQ but only have {BATCH, HIDDEN}
])
```

### Principle 4: Pattern Strings (Alternative Notation)

Instead of `requires/removes/adds`, use einops-style patterns:

```python
class SelectLayer(Op):
    pattern = "layer ... -> ..."  # Removes layer, keeps rest

class PoolSequence(Op):
    pattern = "... seq hidden -> ... hidden"  # Removes seq

class LinearClassify(Op):
    pattern = "... hidden -> ... class"  # Broadcasts over ...

class AttentionClassify(Op):
    pattern = "batch seq hidden -> batch class"  # Exact dims required
```

The `...` means "any additional dimensions (preserved)".

---

## Proposed Module Structure

```
probelab/
├── __init__.py
├── types.py              # Axis, Message, Dialogue, Label
│
├── tensor.py             # AxisTensor - unified container
│   - AxisTensor          # Tensor with axis tracking
│   - axes property       # Current axes tuple
│   - has_axis()          # Check if axis present
│   - shape_of(axis)      # Get size of axis
│
├── datasets/             # Unchanged
│   ├── base.py           # Dataset base class
│   ├── registry.py       # load(), list_datasets()
│   └── *.py              # Domain-specific datasets
│
├── masks/                # Unchanged
│   └── *.py              # Composable mask functions
│
├── tokenize.py           # Single file for tokenization
│   - tokenize()          # Dialogues → TokenizedBatch
│   - TokenizedBatch      # Container for tokenized inputs
│
├── collect.py            # Single file for activation collection
│   - collect()           # Model + Tokens → AxisTensor
│   - HookedModel         # Internal hook management
│
├── ops/                  # All tensor operations
│   ├── __init__.py       # Export all ops
│   ├── base.py           # Op base class with contracts
│   ├── select.py         # SelectLayer, SelectLayers
│   ├── pool.py           # Pool (unified for any dim)
│   ├── normalize.py      # Normalize
│   ├── linear.py         # LinearClassify (Logistic)
│   ├── mlp.py            # MLPClassify
│   └── attention.py      # AttentionClassify
│
├── pipeline.py           # Compose ops with validation
│
├── metrics.py            # Evaluation functions
│
└── viz.py                # Visualization utilities
```

### Key Changes from Current

1. **`preprocessing/` + `probes/` → `ops/`**: Unify all tensor operations
2. **`processing/` split**: `activations.py` → `tensor.py` + `collect.py`
3. **Explicit dimension contracts**: Every op declares requires/removes/adds
4. **Pipeline validation**: Check dimension flow at construction

---

## The AxisTensor Abstraction

Core container for both activations and scores:

```python
@dataclass
class AxisTensor:
    """Tensor with explicit axis tracking."""
    data: torch.Tensor
    axes: tuple[Axis, ...]

    # Optional metadata
    layer_indices: tuple[int, ...] | None = None
    detection_mask: torch.Tensor | None = None

    def has_axis(self, axis: Axis) -> bool:
        return axis in self.axes

    def axis_dim(self, axis: Axis) -> int:
        """Get the dimension index for an axis."""
        return self.axes.index(axis)

    def axis_size(self, axis: Axis) -> int:
        """Get the size of an axis."""
        return self.data.shape[self.axis_dim(axis)]

    # Operations return new AxisTensor with updated axes
    def index_select(self, axis: Axis, index: int) -> "AxisTensor":
        """Select along axis, removing it."""
        dim = self.axis_dim(axis)
        new_data = self.data.select(dim, index)
        new_axes = tuple(a for a in self.axes if a != axis)
        return AxisTensor(new_data, new_axes, ...)

    def reduce(self, axis: Axis, method: str) -> "AxisTensor":
        """Reduce along axis, removing it."""
        dim = self.axis_dim(axis)
        if method == "mean":
            new_data = self.data.mean(dim)
        elif method == "max":
            new_data = self.data.max(dim).values
        # ...
        new_axes = tuple(a for a in self.axes if a != axis)
        return AxisTensor(new_data, new_axes, ...)
```

### Activations vs Scores

Both use `AxisTensor`, distinguished by which axes they have:

```python
# Activations: have HIDDEN axis
activations = AxisTensor(data, axes=(LAYER, BATCH, SEQ, HIDDEN))

# Scores: have CLASS axis instead of HIDDEN
scores = AxisTensor(data, axes=(BATCH, SEQ, CLASS))
# or
scores = AxisTensor(data, axes=(BATCH, CLASS))
```

Could optionally have subclasses for type safety:

```python
class Activations(AxisTensor):
    """AxisTensor that must have HIDDEN axis."""
    def __post_init__(self):
        assert HIDDEN in self.axes

class Scores(AxisTensor):
    """AxisTensor that must have CLASS axis."""
    def __post_init__(self):
        assert CLASS in self.axes
```

---

## Op Examples in Detail

### SelectLayer

```python
class SelectLayer(Op):
    """Select a single layer, removing LAYER axis."""

    requires = (LAYER,)
    removes = (LAYER,)
    adds = ()
    broadcasts = True

    def __init__(self, layer: int):
        self.layer = layer

    def forward(self, x: AxisTensor) -> AxisTensor:
        # Find which position in layer_indices matches
        idx = x.layer_indices.index(self.layer)
        return x.index_select(LAYER, idx)
```

### Pool

```python
class Pool(Op):
    """Pool over a dimension, removing it."""

    def __init__(self, axis: Axis, method: str = "mean"):
        self.axis = axis
        self.method = method

        # Set dimension contract based on axis
        self.requires = (axis,)
        self.removes = (axis,)
        self.adds = ()
        self.broadcasts = True

    def forward(self, x: AxisTensor) -> AxisTensor:
        return x.reduce(self.axis, self.method)
```

### LinearClassify (Logistic)

```python
class LinearClassify(Op):
    """Linear classifier. Replaces HIDDEN with CLASS."""

    requires = (HIDDEN,)
    removes = (HIDDEN,)
    adds = (CLASS,)
    broadcasts = True  # Works on [B,H] or [B,S,H] or [L,B,S,H]

    def __init__(self, hidden_dim: int, device: str = "cuda"):
        self.hidden_dim = hidden_dim
        self.device = device
        self.weight = None
        self.bias = None
        self._fitted = False

    def forward(self, x: AxisTensor) -> AxisTensor:
        # x.data: [..., hidden]
        # output: [..., 2]
        logits = x.data @ self.weight + self.bias
        probs = torch.softmax(logits, dim=-1)

        # Replace HIDDEN with CLASS in axes
        new_axes = tuple(CLASS if a == HIDDEN else a for a in x.axes)
        return AxisTensor(probs, new_axes)

    def fit(self, x: AxisTensor, y: torch.Tensor) -> "LinearClassify":
        # Initialize parameters
        self.weight = torch.zeros(self.hidden_dim, 2, device=self.device)
        self.bias = torch.zeros(2, device=self.device)

        # Training loop...
        ...

        self._fitted = True
        return self
```

### AttentionClassify

```python
class AttentionClassify(Op):
    """Attention-pooled classifier. Reduces SEQ, replaces HIDDEN with CLASS."""

    requires = (BATCH, SEQ, HIDDEN)
    removes = (SEQ, HIDDEN)
    adds = (CLASS,)
    broadcasts = False  # Requires exactly these dims

    def __init__(self, hidden_dim: int, device: str = "cuda"):
        self.hidden_dim = hidden_dim
        self.device = device
        # Attention + classification parameters
        self.query = None
        self.classifier = None
        self._fitted = False

    def forward(self, x: AxisTensor) -> AxisTensor:
        # x.data: [batch, seq, hidden]

        # Compute attention weights
        scores = (x.data @ self.query).squeeze(-1)  # [batch, seq]
        if x.detection_mask is not None:
            scores = scores.masked_fill(~x.detection_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)  # [batch, seq]

        # Weighted sum over sequence
        pooled = (x.data * weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]

        # Classify
        logits = pooled @ self.classifier
        probs = torch.softmax(logits, dim=-1)  # [batch, 2]

        return AxisTensor(probs, axes=(BATCH, CLASS))

    def fit(self, x: AxisTensor, y: torch.Tensor) -> "AttentionClassify":
        # Training...
        self._fitted = True
        return self
```

---

## Pipeline with Validation

```python
class Pipeline:
    def __init__(self, ops: list[Op]):
        self.ops = ops
        self._validate_dimension_flow()

    def _validate_dimension_flow(self):
        """Validate that ops are dimensionally compatible."""
        # Start with full activation axes
        current_axes = {LAYER, BATCH, SEQ, HIDDEN}

        for i, op in enumerate(self.ops):
            # Check requirements
            missing = set(op.requires) - current_axes
            if missing and not op.broadcasts:
                raise ValueError(
                    f"Op {i} ({type(op).__name__}) requires {op.requires} "
                    f"but current axes are {current_axes}. Missing: {missing}"
                )

            # For broadcasting ops, just check that required axes exist
            if op.broadcasts:
                for axis in op.requires:
                    if axis not in current_axes:
                        raise ValueError(
                            f"Op {i} ({type(op).__name__}) requires {axis} "
                            f"but current axes are {current_axes}"
                        )

            # Apply transformation
            current_axes -= set(op.removes)
            current_axes |= set(op.adds)

        # Validate final state
        if CLASS not in current_axes:
            raise ValueError(
                f"Pipeline must produce CLASS axis, but final axes are {current_axes}"
            )
        if BATCH not in current_axes:
            raise ValueError(
                f"Pipeline must preserve BATCH axis, but final axes are {current_axes}"
            )

    def fit(self, x: AxisTensor, y: torch.Tensor) -> "Pipeline":
        for op in self.ops:
            if hasattr(op, 'fit') and not op._fitted:
                op.fit(x, y)
            x = op.forward(x)
        return self

    def __call__(self, x: AxisTensor) -> AxisTensor:
        for op in self.ops:
            x = op.forward(x)
        return x
```

### Example Pipelines

```python
# Valid: sequence-level training
pipeline = Pipeline([
    SelectLayer(16),        # [L,B,S,H] → [B,S,H]
    Pool(SEQ, "mean"),      # [B,S,H] → [B,H]
    LinearClassify(4096),   # [B,H] → [B,C]
])

# Valid: token-level training with score aggregation
pipeline = Pipeline([
    SelectLayer(16),        # [L,B,S,H] → [B,S,H]
    LinearClassify(4096),   # [B,S,H] → [B,S,C] (broadcasts)
    Pool(SEQ, "mean"),      # [B,S,C] → [B,C]
])

# Valid: attention-based
pipeline = Pipeline([
    SelectLayer(16),        # [L,B,S,H] → [B,S,H]
    AttentionClassify(4096),# [B,S,H] → [B,C]
])

# INVALID: AttentionClassify requires SEQ
pipeline = Pipeline([
    SelectLayer(16),        # [L,B,S,H] → [B,S,H]
    Pool(SEQ, "mean"),      # [B,S,H] → [B,H]
    AttentionClassify(4096),# ERROR: requires SEQ
])
# Raises: "Op 2 (AttentionClassify) requires (BATCH, SEQ, HIDDEN)
#          but current axes are {BATCH, HIDDEN}. Missing: {SEQ}"
```

---

## Summary: Key Design Decisions

1. **Unify transforms and classifiers as "Ops"**
   - All ops have `forward(x) -> x`
   - Stateful ops also have `fit(x, y) -> self`
   - Dimension contracts via `requires/removes/adds`

2. **AxisTensor as unified container**
   - Replaces both Activations and Scores
   - Distinguished by which axes present (HIDDEN vs CLASS)
   - All axis tracking in one place

3. **Pipeline validates dimension flow**
   - Catches invalid compositions at construction time
   - Clear error messages pointing to the problem

4. **Explicit broadcasts**
   - `broadcasts=True` means "works with extra dimensions"
   - LinearClassify: `[..., H] → [..., C]`
   - AttentionClassify: `[B, S, H] → [B, C]` (exact)

5. **Module consolidation**
   - `ops/` contains all tensor operations
   - `collect.py` for activation extraction
   - `tokenize.py` for tokenization
   - Fewer, more focused files

---

## Trade-offs

### Pros
- Dimension semantics are explicit and checkable
- Pipeline validation catches errors early
- Unified Op interface simplifies mental model
- Clear what each op does to tensor shape

### Cons
- More verbose (dimension contracts on every op)
- Breaking change from current API
- Might be over-engineering for simple cases

### Middle Ground
Keep current API but add optional dimension contracts as documentation:

```python
class Logistic(BaseProbe):
    """
    Linear classifier.

    Dimension Contract:
        requires: HIDDEN
        removes: HIDDEN
        adds: CLASS
        broadcasts: True (works with [B,H] or [B,S,H])
    """
```

This documents the behavior without enforcing it in code.
