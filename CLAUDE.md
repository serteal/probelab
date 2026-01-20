# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Information

- **Name**: probelab
- **Version**: 0.1.0
- **Author**: Alex Serrano
- **Python**: >=3.11
- **Build System**: hatchling (modern Python packaging)
- **License**: Apache-2.0

## Core Dependencies

- **PyTorch**: Deep learning framework (v2.7.1+)
- **Transformers**: HuggingFace models (v4.53.0+)
- **scikit-learn**: ML utilities (v1.7.0+)
- **jaxtyping**: Type annotations for arrays
- **einops**: Tensor operations
- **accelerate**: GPU/TPU optimization

## Main API Imports

```python
# Core types
from probelab import Message, Dialogue, Label

# Activation handling
from probelab import HookedModel, Activations, collect_activations

# Pipeline and preprocessing
from probelab import Pipeline
from probelab import preprocessing
from probelab.preprocessing import SelectLayer, SelectLayers, Pool, Normalize

# Probes (used within pipelines)
from probelab.probes import BaseProbe, Logistic, MLP, Attention

# High-level workflows
from probelab.scripts import (
    train_pipelines, evaluate_pipelines,  # Core: work with pre-collected activations
    train_from_model, evaluate_from_model,  # Convenience: one-step from model
)

# Datasets
from probelab.datasets import DialogueDataset, CircuitBreakersDataset, DolusChatDataset

# Metrics
from probelab.metrics import auroc, recall_at_fpr, get_metric_by_name

# Visualization
from probelab.visualization import print_metrics, visualize_mask

# Masks for selective token processing
from probelab import masks
```

## Commands

### Development Setup

```bash
# Clone and install in development mode
git clone <repo_url>
cd probelab
uv sync --dev

# Verify installation
uv run python -c "import probelab; print(probelab.__version__)"
```

### Testing

```bash
# Run all tests
uv run pytest tests/

# Run tests with verbose output
uv run pytest tests/ -v

# Run specific test modules
uv run pytest tests/processing/ -v  # All processing tests
uv run pytest tests/probes/ -v      # All probe tests
uv run pytest tests/datasets/ -v    # All dataset tests

# Run specific test files
uv run pytest tests/processing/test_tokenization.py -v
uv run pytest tests/processing/test_activations.py -v
uv run pytest tests/probes/test_logistic.py -v
uv run pytest tests/probes/test_mlp.py -v

# Run tests for specific model
uv run pytest tests/ -k "llama" -v
uv run pytest tests/ -k "gemma" -v

# Run tests by marker
uv run pytest tests/ -m "not slow"      # Exclude slow tests
uv run pytest tests/ -m "requires_gpu"  # GPU tests only

# Run with coverage
uv run pytest tests/ --cov=probelab --cov-report=html --cov-report=term
```

### Code Quality

```bash
# Run linting (configured to ignore F722 for jaxtyping)
uv run ruff check probelab/
uv run ruff format probelab/

# Type checking (if mypy is added)
# uv run mypy src/probelab
```

### Build and Distribution

```bash
# Build the package
uv build

# The build creates:
# - dist/probelab-0.1.0-py3-none-any.whl
# - dist/probelab-0.1.0.tar.gz

# Install from wheel
pip install dist/probelab-0.1.0-py3-none-any.whl
```

### Development Dependencies

The `dev` dependency group includes:

- `huggingface-hub`: For downloading models and datasets from HF Hub
- `ipykernel`: Jupyter notebook kernel for interactive development
- `ipywidgets`: Interactive widgets for notebooks
- `python-dotenv`: Environment variable management (.env files)

## Architecture Overview

probelab is a library for training classifiers (probes) on Large Language Model (LLM) activations to understand what information is encoded in different model layers. It's designed for interpretability research, particularly for understanding how LLMs represent concepts internally.

### Core Concepts

1. **Probes**: Classifiers trained on LLM activations to detect specific properties
2. **Dialogues**: Structured conversations with detection flags for training
3. **Activations**: Hidden states extracted from LLM layers during forward passes
4. **Detection Masks**: Binary masks indicating which tokens to use for training

### Module Structure

```
probelab/
├── __init__.py          # Public API exports
├── types.py             # Core type definitions
├── pipeline.py          # Pipeline composition for preprocessing + probes
├── datasets/            # Dataset handling
│   ├── base.py         # DialogueDataset base class
│   ├── deception.py    # Deception detection datasets
│   ├── harmfulness.py  # Harmfulness detection datasets
│   └── ood.py          # Out-of-distribution datasets
├── models/              # Model interfaces
│   ├── architectures.py # Model-specific configurations
│   └── hooks.py        # PyTorch hook management
├── processing/          # Data processing
│   ├── activations.py  # Activation extraction and containers
│   ├── scores.py       # Score containers for predictions
│   └── tokenization.py # Dialogue tokenization
├── preprocessing/       # Preprocessing transformers
│   ├── base.py         # PreTransformer base class
│   └── pre_transforms.py # SelectLayer, SelectLayers, Pool, Normalize
├── probes/              # Probe implementations
│   ├── base.py         # BaseProbe abstract class
│   ├── logistic.py     # Logistic regression probes
│   ├── mlp.py          # MLP probe
│   └── attention.py    # Attention-based probe
├── masks/               # Mask functions for selective token processing
│   ├── base.py         # MaskFunction base class
│   ├── basic.py        # Basic masks (all, none, last_token, etc.)
│   ├── role.py         # Role-based masks (assistant, user, system)
│   ├── text.py         # Text matching masks (contains, regex)
│   ├── position.py     # Position masks (between, after, before, nth_message)
│   ├── content.py      # Content masks (special_tokens, padding)
│   └── composite.py    # Composite masks (AndMask, OrMask, NotMask)
├── scripts/             # High-level workflows
│   └── workflows.py    # train_pipelines, evaluate_pipelines, convenience functions
├── metrics.py           # Function-based metrics API
├── visualization.py     # Plotting and visualization
└── logger.py           # Logging configuration
```

### Core Components

1. **Types (`types.py`)**: Fundamental data structures

   - `Message`: Single dialogue turn with role, content, and detect flag
   - `Dialogue`: List of Messages representing a conversation
   - `Label`: Enum for binary classification (NEGATIVE=0, POSITIVE=1)
   - `Role`: Literal type for message roles ("system", "user", "assistant")

2. **Datasets (`datasets/`)**: Data loading and management

   - `DialogueDataset`: Abstract base class for all datasets
   - Supports slicing, filtering, concatenation, and metadata
   - Model-specific padding configurations
   - Hash-based caching for reproducibility
   - **Deception datasets**: AIAuditDataset, AILiarDataset, DolusChatDataset, REPEDataset,
     SandbaggingDataset, TruthfulQADataset, WerewolfDataset, RoleplayingDataset,
     InsiderTradingDataset
   - **Harmfulness datasets**: CircuitBreakersDataset, BenignInstructionsDataset,
     WildJailbreakDataset, WildGuardMixDataset, XSTestResponseDataset, CoconotDataset,
     ToxicChatDataset, ClearHarmLlama3Dataset, ClearHarmMistralSmallDataset
   - **OOD datasets**: AlpacaDataset, LmsysChatDataset, MATHInstructionDataset,
     UltraChatDataset, AlpacaFrenchDataset, GenericSpanishDataset, and more

3. **Models (`models/`)**: LLM interfaces

   - `HookedModel`: Context manager for activation extraction with `hook_point` control
   - Hook points:
     - `"post_block"` (default): After attention + MLP, after final layernorm
     - `"pre_layernorm"`: Before the initial layernorm in each layer
   - Architecture-specific handlers (LLaMA, Gemma, etc.)
   - Automatic layer detection and hook management
   - Memory-efficient activation collection

4. **Processing (`processing/`)**: Data transformation pipeline

   - `collect_activations`: Extract hidden states from models (main API)
   - `Activations`: Axis-aware container class with methods:
     - `from_tensor()`: Create from pre-stacked 4D tensor [layer, batch, seq, hidden]
     - `from_hidden_states()`: Create from HuggingFace nested tuple format or tensor
     - `pool()`: Unified pooling over sequence or layer dimension (mean, max, last_token)
     - `select(layer=int)`: Select single layer, removes LAYER axis
     - `select(layers=list)`: Select multiple layers, keeps LAYER axis
     - `to()`: Device/dtype conversion
     - Properties: `n_layers`, `batch_size`, `seq_len`, `d_model`, `layer_indices`
   - `ActivationIterator`: Memory-efficient streaming wrapper
   - `tokenize_dialogues(tokenizer, dialogues, mask, ...)`: Convert dialogues to model inputs (mask is required)
   - `tokenize_dataset(dataset, tokenizer, mask, ...)`: Batch tokenization (mask is required)

5. **Probes (`probes/`)**: Classifier implementations

   - **BaseProbe**: Abstract base for probe implementations
   - **Logistic**: L2-regularized logistic regression
   - **SklearnLogistic**: Scikit-learn based variant
   - **MLP**: Multi-layer perceptron with dropout
   - **Attention**: Attention-weighted classification
   - Probes no longer handle layer selection or aggregation internally
   - Used within pipelines for composition with preprocessing steps

6. **Pipeline (`pipeline.py`)**: Composition framework

   - **Pipeline**: Compose preprocessing steps with probes
   - Methods: `fit()`, `predict()`, `predict_proba()`
   - **Auto-selection**: Single-layer activations work without `SelectLayer` (auto-selected)
   - Multi-layer activations require explicit handling (`SelectLayer` or `Pool(dim="layer")`)
   - Enables clear separation of concerns
   - Example: `Pipeline([Pool(dim="sequence", method="mean"), Logistic()])` (single layer)
   - Example: `Pipeline([SelectLayer(16), Pool(dim="sequence", method="mean"), Logistic()])` (multi-layer)

7. **Preprocessing (`preprocessing/`)**: Transformation steps

   - **SelectLayer**: Select a single layer from multi-layer activations (removes LAYER axis)
   - **SelectLayers**: Select multiple layers (keeps LAYER axis)
   - **Pool**: Unified pooling over sequence or layer dimension (mean, max, last_token)
   - **Normalize**: Feature normalization (compute mean/std on training data)
   - All transformers follow sklearn-like API (fit, transform, fit_transform)

8. **Scripts (`scripts/workflows.py`)**: High-level workflows

   - **Core functions** (work with pre-collected activations):
     - `train_pipelines`: Train pipelines on activation batches
     - `evaluate_pipelines`: Evaluate pipelines on activation batches
   - **Convenience functions** (one-step from model):
     - `train_from_model`: Collect activations + train pipelines
     - `evaluate_from_model`: Collect activations + evaluate pipelines (supports streaming evaluation)
   - Explicit 2-step API preferred over convenience functions
   - Multi-pipeline training with shared activations

9. **Metrics (`metrics.py`)**: Function-based metrics API

   - Core metrics: `auroc`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`
   - Special metrics: `recall_at_fpr`, `partial_auroc`, `fpr_at_threshold`
   - Bootstrap confidence intervals via `@with_bootstrap()` decorator
   - String-based metric lookup: `get_metric_by_name("auroc")`
   - Parameterized metrics: `"recall@5"` (5% FPR), `"percentile95"`

10. **Visualization (`visualization.py`)**: Result visualization
    - ROC and precision-recall curves
    - Recall comparison bar charts
    - Detection mask visualization with `visualize_mask()`
    - Modern plotting theme with accessibility

11. **Masks (`masks/`)**: Selective token processing

   - **MaskFunction**: Base class for composable mask functions
   - **Basic masks**: `all()`, `none()`, `last_token()`, `first_n_tokens()`, `last_n_tokens()`
   - **Role masks**: `assistant()`, `user()`, `system()`, `role()` for custom roles
   - **Text masks**: `contains()`, `regex()` for pattern matching
   - **Position masks**: `between()`, `after()`, `before()`, `nth_message()`, `padding()`
   - **Content masks**: `special_tokens()` for filtering special tokens
   - **Composite masks**: `AndMask`, `OrMask`, `NotMask` for boolean logic
   - Used with `collect_activations(mask=...)` to control which tokens are detected
   - All masks are composable and chainable

### Key Design Patterns

1. **Axis-Aware Activations**

   - `Activations` tracks which dimensions exist via `axes` tuple
   - Standard axes: `[LAYER, BATCH, SEQ, HIDDEN]`
   - Operations automatically handle axis presence/removal:
     - `select(layers=16)` removes LAYER axis → `[BATCH, SEQ, HIDDEN]`
     - `pool(dim="sequence")` removes SEQ axis → `[LAYER, BATCH, HIDDEN]`
   - Properties adapt to available axes (`n_layers`, `batch_size`, `seq_len`, `d_model`)
   - Enables safe chaining of transformations without shape confusion

2. **Mask-Based Detection Control**

   - Mask functions control which tokens are detected during activation collection
   - Composable via boolean logic (AndMask, OrMask, NotMask)
   - Applied during tokenization, not on dialogues
   - Examples: `pl.masks.assistant()`, `pl.masks.nth_message(-1)`, `pl.masks.contains("yes")`
   - Replaces old `detect` flag on messages for cleaner separation of concerns

3. **Memory Efficiency**

   - Streaming activation collection via `ActivationIterator` (for evaluation)
   - Dynamic batching based on sequence lengths
   - Tensor views for efficient batch processing
   - Optional model pruning (keep only needed layers)
   - Activation caching with deterministic hashing
   - Use `collection_strategy="mean"` for memory-efficient training on large datasets

4. **Pipeline-Based Architecture**

   - Explicit composition of preprocessing steps and probes
   - Clear separation of concerns: layer selection, pooling, classification
   - Pipelines follow sklearn-like API (fit, predict, predict_proba)
   - Enables easy experimentation with different preprocessing strategies
   - Example: `Pipeline([SelectLayer(16), Pool(dim="sequence", method="mean"), Logistic()])`

5. **Unified Interfaces**

   - Core workflow functions for pre-collected activations (preferred)
   - Convenience functions for one-step training from model
   - Unified `pool()` and `select()` methods for activation manipulation
   - Automatic format detection (DialogueDataset, list[Dialogue], HF hidden states)
   - Works with both single pipelines and pipeline dictionaries

6. **Extensibility**
   - Abstract base classes for datasets, probes, masks, and transformers
   - Plugin architecture for new model support
   - Composable preprocessing transformers
   - Custom metrics via string identifiers

### Common Workflows

1. **Basic Pipeline Training (2-Step API - Preferred)**

   ```python
   import probelab as pl

   # Load data
   dataset = pl.datasets.CircuitBreakersDataset()

   # Step 1: Collect activations (single layer)
   activations = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=dataset,
       layers=[12],
       mask=pl.masks.assistant(),
       batch_size=32
   )

   # Step 2: Create pipeline and train
   # NOTE: SelectLayer not needed for single-layer activations (auto-selected)
   pipeline = pl.Pipeline([
       ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
       ("probe", pl.probes.Logistic(device="cuda")),
   ])

   pipeline.fit(activations, dataset.labels)
   ```

2. **Convenience Function (One-Step from Model)**

   ```python
   import probelab as pl

   # Load data
   dataset = pl.datasets.CircuitBreakersDataset()

   # Create pipeline (SelectLayer not needed for single layer)
   pipeline = pl.Pipeline([
       ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
       ("probe", pl.probes.Logistic(device="cuda")),
   ])

   # One-step training (collects activations internally)
   pl.scripts.train_from_model(
       pipeline, model, tokenizer, dataset,
       layers=[12],  # required for convenience functions
       labels=dataset.labels,
       mask=pl.masks.assistant(),
       batch_size=32
   )
   ```

3. **Multi-Layer Analysis**

   ```python
   # Collect activations from multiple layers
   activations = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=dataset,
       layers=[8, 12, 16, 20],
       mask=pl.masks.assistant(),
       batch_size=32
   )

   # Train pipelines on different layers
   pipelines = {
       f"layer_{i}": pl.Pipeline([
           ("select", pl.preprocessing.SelectLayer(i)),
           ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
           ("probe", pl.probes.Logistic(device="cuda")),
       ])
       for i in [8, 12, 16, 20]
   }

   pl.scripts.train_pipelines(pipelines, activations, dataset.labels)
   ```

4. **Memory-Efficient Training for Large Datasets**
   ```python
   # Use collection_strategy to pool activations during collection
   # This reduces memory ~440x by avoiding storing full sequences
   activations = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=large_dataset,
       layers=[12],
       mask=pl.masks.assistant(),
       batch_size=8,
       collection_strategy="mean"  # Pool during collection
   )

   # Create pipeline (no Pool needed - already aggregated)
   pipeline = pl.Pipeline([
       ("select", pl.preprocessing.SelectLayer(12)),
       ("probe", pl.probes.Logistic(device="cuda")),
   ])

   pipeline.fit(activations, large_dataset.labels)
   ```

5. **Evaluation with Custom Metrics**
   ```python
   import functools

   # Collect test activations
   test_acts = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=test_data,
       layers=[12],
       mask=pl.masks.assistant(),
       batch_size=32
   )

   # Evaluate with custom metrics
   predictions, metrics = pl.scripts.evaluate_pipelines(
       pipeline, test_acts,
       labels=test_labels,
       metrics=[
           "auroc",
           "balanced_accuracy",
           functools.partial(pl.metrics.recall_at_fpr, fpr=0.01)
       ]
   )
   ```

6. **Using Masks for Fine-Grained Control**
   ```python
   # Only detect on last assistant message
   mask = pl.masks.AndMask(
       pl.masks.assistant(),
       pl.masks.nth_message(-1)  # Last message
   )

   acts = pl.collect_activations(
       data=dataset,
       model=model,
       tokenizer=tokenizer,
       layers=[16],
       mask=mask,
       batch_size=32
   )

   # Create and train pipeline
   pipeline = pl.Pipeline([
       ("select", pl.preprocessing.SelectLayer(16)),
       ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
       ("probe", pl.probes.Logistic(device="cuda")),
   ])
   pipeline.fit(acts, dataset.labels)
   ```

7. **Token-Level Prediction with Score Aggregation**
   ```python
   # Collect activations
   acts = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=dataset,
       layers=[16],
       mask=pl.masks.assistant(),
       batch_size=32
   )

   # Pipeline that trains on tokens, then aggregates scores
   pipeline = pl.Pipeline([
       ("select", pl.preprocessing.SelectLayer(16)),
       ("probe", pl.probes.Logistic(device="cuda")),  # Trains on individual tokens
       ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),  # Aggregates predictions
   ])
   pipeline.fit(acts, dataset.labels)
   ```

8. **Manual Activation Manipulation**
   ```python
   # Collect activations
   acts = pl.collect_activations(
       data=dataset,
       model=model,
       tokenizer=tokenizer,
       layers=[8, 16, 24],
       mask=pl.masks.assistant(),
       batch_size=32
   )

   # Select specific layer
   layer_16 = acts.select(layer=16)  # Removes LAYER axis

   # Pool over sequence
   pooled = acts.pool(dim="sequence", method="mean")  # [layers, batch, hidden]

   # Select multiple layers
   mid_layers = acts.select(layers=[8, 16])  # Keeps LAYER axis
   ```

9. **Using Different Hook Points**
    ```python
    # Extract activations before layernorm (earlier in computation)
    acts_pre = pl.collect_activations(
        data=dataset,
        model=model,
        tokenizer=tokenizer,
        layers=[16],
        mask=pl.masks.assistant(),
        hook_point="pre_layernorm",
        batch_size=32
    )

    # Extract activations after block (default, post-layernorm)
    acts_post = pl.collect_activations(
        data=dataset,
        model=model,
        tokenizer=tokenizer,
        layers=[16],
        mask=pl.masks.assistant(),
        hook_point="post_block",
        batch_size=32
    )
    ```

### Adding New Features

**New Probe Types:**

1. Create class inheriting from `BaseProbe` in `probes/`
2. Implement required methods: `fit`, `predict`, `predict_proba`
3. Probes no longer handle layer selection or aggregation (done in pipeline)
4. Add comprehensive tests in `tests/probes/`
5. Update documentation

**New Preprocessing Transformers:**

1. Create class inheriting from `PreTransformer` in `preprocessing/pre_transforms.py`
2. Implement required methods: `fit`, `transform`, `fit_transform`
3. Handle activation shape transformations appropriately
4. Add tests in `tests/preprocessing/`
5. Document in this file

**New Model Support:**

1. Add model config to `models/architectures.py`
2. Implement activation extraction in `models/hooks.py`
3. Add tokenization patterns in `processing/tokenization.py`
4. Add padding config to `datasets/base.py`
5. Test with existing probes

**New Datasets:**

1. Inherit from `DialogueDataset` in `datasets/`
2. Implement `_get_dialogues` method
3. Set appropriate `base_name` class attribute
4. Add any dataset-specific post-processing
5. Consider adding metadata support

**New Metrics:**

1. Add metric function to `metrics.py`
2. Register in `METRICS` dictionary
3. Support both tensor and numpy inputs
4. Add bootstrap version if applicable
5. Include in default metrics if commonly used

### Best Practices

1. **Memory Management**

   - Use `collection_strategy="mean"` for large datasets
   - Enable model pruning when using few layers
   - Clear GPU cache between large operations
   - Use appropriate batch sizes for your GPU

2. **Reproducibility**

   - Set random seeds via `random_state` parameters
   - Use dataset hashing for cache keys
   - Log all hyperparameters
   - Save probe configurations with models

3. **Testing**

   - Test with both GPU and CPU
   - Include edge cases (empty datasets, single examples)
   - Verify numerical stability

4. **Performance**
   - Profile activation collection bottlenecks
   - Use TorchScript for production inference
   - Consider quantization for large models
   - Batch similar-length sequences together

### Common Issues and Solutions

1. **CUDA Out of Memory**

   - Reduce batch_size
   - Enable streaming mode
   - Use model pruning
   - Clear cache with torch.cuda.empty_cache()

2. **Tokenization Mismatches**

   - Check model family in tokenizer.name_or_path
   - Verify padding configuration
   - Ensure consistent chat templates
   - Test with `visualize_mask()` to see detection masks
   - Use appropriate mask functions to control detection

3. **Poor Probe Performance**

   - Check class balance in dataset
   - Verify detection masks are correct
   - Try different aggregation methods
   - Increase regularization for logistic probe

4. **Slow Training**
   - Use `collection_strategy="mean"` to reduce memory and improve throughput
   - Use larger batch sizes if memory allows
   - Consider using fewer layers
   - Profile with PyTorch profiler

### Environment Variables

```bash
# Logging level
PROBELAB_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Cache directory
PROBELAB_CACHE_DIR=/path/to/cache

# Default device
PROBELAB_DEVICE=cuda:0  # or cpu

# Disable progress bars
PROBELAB_DISABLE_PROGRESS=1
```

### Pytest Configuration

Located in `pyproject.toml`:

- Test discovery: `tests/` directory
- Naming convention: `test_*.py` files
- Custom markers: `slow`, `requires_gpu`
- Coverage targets: 80%+ recommended

### Recent Updates & Future Enhancements

**Recently Added (Latest):**
- **Removed streaming training** (simplification):
  - Removed `pipeline.partial_fit()`, `pipeline.fit_streaming()`
  - Removed `train_pipelines_streaming()` function
  - Removed `streaming` parameter from `train_from_model()`
  - Removed all probe `partial_fit()` methods
  - **Migration:** Use batch training. For large datasets, use `collection_strategy="mean"` in `collect_activations()` for memory efficiency
  - Streaming evaluation is still supported via `evaluate_from_model(streaming=True)`
- **API refinements for research clarity:**
  - `tokenize_dialogues()` and `tokenize_dataset()` now require `mask` parameter (no hidden defaults)
  - `Activations.select()` now uses explicit `layer=` (single, removes axis) and `layers=` (multiple, keeps axis) parameters
  - Removed ambiguous `layers=int` overloading that automatically detected cardinality
- **Pipeline-based API** (Breaking changes, no backward compatibility):
  - `Pipeline`: Explicit composition of preprocessing steps and probes
  - Preprocessing transformers: `SelectLayer`, `SelectLayers`, `Pool`, `Normalize`
  - Workflow functions renamed:
    - Core: `train_pipelines()`, `evaluate_pipelines()` (work with pre-collected activations)
    - Convenience: `train_from_model()`, `evaluate_from_model()` (one-step from model)
  - Probes no longer have `layer`, `sequence_pooling`, or `score_aggregation` parameters
  - All preprocessing is now explicit in pipeline definition
  - Clear separation of concerns: layer selection, aggregation, classification

**Removed (Breaking Changes):**
- **Streaming training** (latest):
  - `pipeline.partial_fit()` → Use `pipeline.fit()` with batch data
  - `pipeline.fit_streaming()` → Use `pipeline.fit()` with batch data
  - `train_pipelines_streaming()` → Use `train_pipelines()` with batch data
  - `streaming` parameter in `train_from_model()` → Use batch training
  - All probe `partial_fit()` methods → Use `probe.fit()` with batch data
  - `Normalize.partial_fit()` → Use `Normalize.fit()` with batch data
  - `Logistic.streaming_lr` parameter → Removed (no streaming training)
- **Probe parameters**:
  - `layer` parameter → Use `SelectLayer` transformer in pipeline
  - `sequence_pooling` parameter → Use `Pool(dim="sequence", ...)` transformer in pipeline
  - `score_aggregation` parameter → Use `Pool(dim="sequence", ...)` transformer after probe in pipeline
  - `SequencePooling` enum → Use string values in `Pool` ("mean", "max", "last_token")
- **Workflow functions**:
  - `train_probes()` → Use `train_pipelines()` or `train_from_model()`
  - `evaluate_probes()` → Use `evaluate_pipelines()` or `evaluate_from_model()`
- **Activation methods** (from previous refactoring):
  - `Activations.aggregate()` → Use `Activations.pool(dim="sequence")`
  - `Activations.sequence_pool()` → Use `Activations.pool(dim="sequence")`
  - `Activations.select_layer()` → Use `Activations.select(layer=...)`
  - `Activations.select_layers()` → Use `Activations.select(layers=[...])`

**API Migration Guide:**
```python
# OLD API (removed)
probe = pl.probes.Logistic(
    layer=16,
    sequence_pooling=SequencePooling.MEAN,
    device="cuda"
)
pl.scripts.train_probes(
    probe, model, tokenizer, dataset,
    labels=dataset.labels,
    mask=pl.masks.assistant()
)

# NEW API (current) - 2-step approach (preferred)
# Step 1: Collect activations
acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=dataset,
    layers=[16],
    mask=pl.masks.assistant(),
    batch_size=32
)

# Step 2: Create pipeline and train
pipeline = pl.Pipeline([
    ("select", pl.preprocessing.SelectLayer(16)),
    ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
    ("probe", pl.probes.Logistic(device="cuda")),
])
pipeline.fit(acts, dataset.labels)

# OR convenience function (one-step)
pl.scripts.train_from_model(
    pipeline, model, tokenizer, dataset,
    layers=[16],  # required for convenience functions
    labels=dataset.labels,
    mask=pl.masks.assistant()
)
```

**Previous Updates:**
- **Unified Activations API**:
  - `Activations.pool()`: Unified method for pooling
  - `Activations.select()`: Unified method for layer selection
  - `Activations.from_hidden_states()`: Create from HuggingFace format
- **Mask system**: Comprehensive composable mask functions
- Function-based metrics API with bootstrap confidence intervals
- `ActivationIterator` for streaming evaluation
- Multi-pipeline parallel training with shared activations
- Attention probe implementation

**Future Enhancements:**
- Multi-dataset train/eval (combine datasets for training)
- TorchScript compilation for production inference
- Additional preprocessing transformers (PCA, etc.)
- Additional mask functions (token-level pattern matching, custom predicates)
- Integration utilities for external frameworks (inspect_ai, control-arena)
