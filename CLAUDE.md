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

# Pipeline and transforms
from probelab import Pipeline
from probelab.transforms import pre, post  # Type-safe transform modules
# Pre-probe transforms (Activations → Activations)
# pre.SelectLayer, pre.SelectLayers, pre.Pool, pre.Normalize
# Post-probe transforms (Scores → Scores)
# post.Pool, post.EMAPool, post.RollingPool

# Probes (used within pipelines)
from probelab.probes import BaseProbe, Logistic, MLP, Attention

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
├── transforms/          # Transform modules (type-safe)
│   ├── base.py         # ActivationTransform, ScoreTransform base classes
│   ├── pre.py          # Pre-probe: SelectLayer, SelectLayers, Pool, Normalize
│   └── post.py         # Post-probe: Pool, EMAPool, RollingPool
├── utils/               # Internal utilities
│   └── validation.py   # check_activations(), check_scores() (internal)
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
├── metrics.py           # Function-based metrics API
├── visualization.py     # Plotting and visualization
├── logger.py           # Logging configuration
└── types.py            # Core type definitions
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

   - **Pipeline**: Compose pre-transforms, probe, and post-transforms
   - Methods: `fit()`, `predict()` (returns probabilities [batch, 2]), `score()`
   - Pre-transforms must be `ActivationTransform` (before probe)
   - Post-transforms must be `ScoreTransform` (after probe)
   - Example: `Pipeline([("select", pre.SelectLayer(16)), ("pool", pre.Pool(dim="sequence")), ("probe", Logistic())])`

7. **Transforms (`transforms/`)**: Type-safe transformation steps

   - **`transforms.pre`** - Pre-probe transforms (Activations → Activations):
     - `SelectLayer`: Select single layer (removes LAYER axis)
     - `SelectLayers`: Select multiple layers (keeps LAYER axis)
     - `Pool`: Pool over sequence or layer dimension
     - `Normalize`: Feature normalization
   - **`transforms.post`** - Post-probe transforms (Scores → Scores):
     - `Pool`: Aggregate token scores to sequence level
     - `EMAPool`: Exponential moving average aggregation
     - `RollingPool`: Rolling window aggregation
   - All follow sklearn-like API (fit, transform, fit_transform)

8. **Metrics (`metrics.py`)**: Function-based metrics API

   - Core metrics: `auroc`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`
   - Special metrics: `recall_at_fpr`, `partial_auroc`, `fpr_at_threshold`
   - Bootstrap confidence intervals via `@with_bootstrap()` decorator
   - String-based metric lookup: `get_metric_by_name("auroc")`
   - Parameterized metrics: `"recall@5"` (5% FPR), `"percentile95"`

9. **Visualization (`visualization.py`)**: Result visualization
    - ROC and precision-recall curves
    - Recall comparison bar charts
    - Detection mask visualization with `visualize_mask()`
    - Modern plotting theme with accessibility

10. **Masks (`masks/`)**: Selective token processing

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

   - Streaming activation collection via `ActivationIterator`
   - Automatic streaming detection based on dataset size
   - Dynamic batching based on sequence lengths
   - Tensor views for efficient batch processing
   - Optional model pruning (keep only needed layers)
   - Activation caching with deterministic hashing
   - `partial_fit()` support for incremental training

4. **Pipeline-Based Architecture**

   - Explicit composition of preprocessing steps and probes
   - Clear separation of concerns: layer selection, pooling, classification
   - Pipelines follow sklearn-like API (fit, predict, partial_fit)
   - Enables easy experimentation with different preprocessing strategies
   - Example: `Pipeline([SelectLayer(16), Pool(dim="sequence", method="mean"), Logistic()])`

5. **Unified Interfaces**

   - Unified `pool()` and `select()` methods for activation manipulation
   - Automatic format detection (DialogueDataset, list[Dialogue], HF hidden states)
   - Direct use of `pipeline.fit()` and `pipeline.predict_proba()` for training and inference

6. **Extensibility**
   - Abstract base classes for datasets, probes, masks, and transformers
   - Plugin architecture for new model support
   - Composable preprocessing transformers
   - Custom metrics via string identifiers

### Common Workflows

1. **Basic Pipeline Training (2-Step API - Preferred)**

   ```python
   import probelab as pl
   from probelab.transforms import pre

   # Load data
   dataset = pl.datasets.CircuitBreakersDataset()

   # Step 1: Collect activations
   activations = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=dataset,
       layers=[12],
       mask=pl.masks.assistant(),
       batch_size=32
   )

   # Step 2: Create pipeline and train
   pipeline = pl.Pipeline([
       ("select", pre.SelectLayer(12)),
       ("pool", pre.Pool(dim="sequence", method="mean")),
       ("probe", pl.probes.Logistic(device="cuda")),
   ])

   pipeline.fit(activations, dataset.labels)
   ```

2. **Multi-Layer Analysis**

   ```python
   from probelab.transforms import pre

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
           ("select", pre.SelectLayer(i)),
           ("pool", pre.Pool(dim="sequence", method="mean")),
           ("probe", pl.probes.Logistic(device="cuda")),
       ])
       for i in [8, 12, 16, 20]
   }

   # Train each pipeline
   for name, pipeline in pipelines.items():
       pipeline.fit(activations, dataset.labels)
   ```

3. **Streaming for Large Datasets**
   ```python
   from probelab.transforms import pre

   # Collect activations in streaming mode
   activation_stream = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=large_dataset,
       layers=[12],
       mask=pl.masks.assistant(),
       batch_size=8,
       streaming=True  # Returns ActivationIterator
   )

   # Create pipeline
   pipeline = pl.Pipeline([
       ("select", pre.SelectLayer(12)),
       ("pool", pre.Pool(dim="sequence", method="mean")),
       ("probe", pl.probes.Logistic(device="cuda")),
   ])

   # Streaming training with pipeline.fit_streaming()
   pipeline.fit_streaming(activation_stream, labels)
   ```

4. **Evaluation with Metrics**
   ```python
   # Collect test activations
   test_acts = pl.collect_activations(
       model=model,
       tokenizer=tokenizer,
       data=test_data,
       layers=[12],
       mask=pl.masks.assistant(),
       batch_size=32
   )

   # Get predictions (predict() returns probabilities [batch, 2])
   probs = pipeline.predict(test_acts)
   y_pred = probs[:, 1].cpu().numpy()
   y_true = [label.value for label in test_labels]

   # Compute metrics
   print(f"AUROC: {pl.metrics.auroc(y_true, y_pred):.3f}")
   print(f"Recall@1%: {pl.metrics.recall_at_fpr(y_true, y_pred, fpr=0.01):.3f}")
   ```

5. **Using Masks for Fine-Grained Control**
   ```python
   from probelab.transforms import pre

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
       ("select", pre.SelectLayer(16)),
       ("pool", pre.Pool(dim="sequence", method="mean")),
       ("probe", pl.probes.Logistic(device="cuda")),
   ])
   pipeline.fit(acts, dataset.labels)
   ```

6. **Token-Level Prediction with Score Aggregation**
   ```python
   from probelab.transforms import pre, post

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
       ("select", pre.SelectLayer(16)),
       ("probe", pl.probes.Logistic(device="cuda")),  # Trains on individual tokens
       ("pool", post.Pool(method="mean")),  # Aggregates predictions (Scores → Scores)
   ])
   pipeline.fit(acts, dataset.labels)
   ```

7. **Manual Activation Manipulation**
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

8. **Using Different Hook Points**
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
2. Implement required methods: `fit`, `predict` (returns Scores), `save`, `load`
3. Probes no longer handle layer selection or aggregation (done in pipeline)
4. Add comprehensive tests in `tests/probes/`
5. Update documentation

**New Pre-Probe Transforms (Activations → Activations):**

1. Create class inheriting from `ActivationTransform` in `transforms/pre.py`
2. Implement required method: `transform(X: Activations) -> Activations`
3. Use `check_activations()` from `probelab.utils.validation` for validation
4. Add tests in `tests/transforms/test_pre.py`
5. Document in this file

**New Post-Probe Transforms (Scores → Scores):**

1. Create class inheriting from `ScoreTransform` in `transforms/post.py`
2. Implement required method: `transform(X: Scores) -> Scores`
3. Use `check_scores()` from `probelab.utils.validation` for validation
4. Add tests in `tests/transforms/test_post.py`
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

   - Use streaming for datasets > 10k examples
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
   - Test streaming and batch modes
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
   - Enable streaming for large datasets
   - Use larger batch sizes if memory allows
   - Consider using fewer layers
   - Profile with PyTorch profiler

### Environment Variables

All configuration is done via environment variables (no config module):

```bash
# Logging level
PROBELAB_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Cache directory
PROBELAB_CACHE_DIR=/path/to/cache

# Default device for probes
PROBELAB_DEFAULT_DEVICE=cuda:0  # or cpu

# Verbose output (progress bars, etc.)
PROBELAB_VERBOSE=true  # or false

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
- **Type-safe transform modules** (Breaking changes):
  - Transforms split into `transforms.pre` (Activations → Activations) and `transforms.post` (Scores → Scores)
  - New base classes: `ActivationTransform` and `ScoreTransform` (replaces `PreTransformer`)
  - Import pattern: `from probelab.transforms import pre, post`
  - Pre-probe: `pre.SelectLayer`, `pre.SelectLayers`, `pre.Pool`, `pre.Normalize`
  - Post-probe: `post.Pool`, `post.EMAPool`, `post.RollingPool`
  - Pipeline validates pre-transforms are `ActivationTransform` and post-transforms are `ScoreTransform`
- **Simplified probe API** (Breaking changes):
  - `predict_proba()` removed from probes and Pipeline
  - `predict()` now returns probabilities `[batch, 2]` (not class labels)
  - Use `probs.argmax(dim=1)` for class labels
- **Centralized validation** (Internal):
  - `check_activations()` and `check_scores()` in `probelab.utils.validation`
  - Validation moved from data classes to transforms (caller validates)
  - Activations/Scores methods no longer raise on missing axes
- **Removed config and profiling modules** (Breaking changes):
  - Removed `config.py` - Context manager, get_config(), set_defaults() all removed
  - Removed `profiling.py` - ProbelabCounters, profile_section() all removed
  - Configuration now done via environment variables only (PROBELAB_*)
  - Simpler, more explicit API with no hidden global state
- **Pipeline-based API**:
  - `Pipeline`: Explicit composition of preprocessing steps and probes
  - Probes no longer have `layer`, `sequence_pooling`, or `score_aggregation` parameters
  - All preprocessing is now explicit in pipeline definition
  - Clear separation of concerns: layer selection, aggregation, classification

**Removed (Breaking Changes):**
- **config module** (entire module removed):
  - `Context` context manager → Use environment variables (PROBELAB_*)
  - `get_config()` → Check environment variables directly
  - `set_defaults()` → Set environment variables
  - `ConfigVar` → Removed (no replacement needed)
- **profiling module** (entire module removed):
  - `ProbelabCounters` → Use external profiling tools (PyTorch profiler, etc.)
  - `profile_section()` → Use Python's built-in profiling or external tools
  - `is_profiling()` → Removed (no replacement needed)
- **scripts module**:
  - `train_pipelines()` → Use `pipeline.fit(activations, labels)`
  - `evaluate_pipelines()` → Use `pipeline.predict_proba(activations)` + manual metrics
  - `train_from_model()` → Removed (use explicit 2-step)
  - `evaluate_from_model()` → Removed (use explicit 2-step)
  - `train_pipelines_streaming()` → Use `pipeline.fit_streaming(activation_iter, labels)`
- **Probe parameters**:
  - `layer` parameter → Use `SelectLayer` transformer in pipeline
  - `sequence_pooling` parameter → Use `Pool(dim="sequence", ...)` transformer in pipeline
  - `score_aggregation` parameter → Use `Pool(dim="sequence", ...)` transformer after probe in pipeline
  - `SequencePooling` enum → Use string values in `Pool` ("mean", "max", "last_token")
- **Activation methods** (from previous refactoring):
  - `Activations.aggregate()` → Use `Activations.pool(dim="sequence")`
  - `Activations.sequence_pool()` → Use `Activations.pool(dim="sequence")`
  - `Activations.select_layer()` → Use `Activations.select(layer=...)`
  - `Activations.select_layers()` → Use `Activations.select(layers=[...])`

**API Migration Guide:**
```python
# OLD imports (removed)
from probelab.preprocessing import SelectLayer, Pool

# NEW imports (current)
from probelab.transforms import pre, post

# OLD API (removed)
pipeline.predict_proba(test_acts)  # Removed method

# NEW API (current)
probs = pipeline.predict(test_acts)  # Returns [batch, 2] probabilities
labels = probs.argmax(dim=1)  # Get class labels

# Full example - direct pipeline methods
# Step 1: Collect activations
acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=dataset,
    layers=[16],
    mask=pl.masks.assistant(),
    batch_size=32
)

# Step 2: Create pipeline and train (use pre.* for pre-probe transforms)
pipeline = pl.Pipeline([
    ("select", pre.SelectLayer(16)),
    ("pool", pre.Pool(dim="sequence", method="mean")),
    ("probe", pl.probes.Logistic(device="cuda")),
])
pipeline.fit(acts, dataset.labels)

# Step 3: Evaluate (predict() returns probabilities)
test_acts = pl.collect_activations(...)
probs = pipeline.predict(test_acts)  # Returns [batch, 2]
y_pred = probs[:, 1].cpu().numpy()
y_true = [label.value for label in test_labels]
print(f"AUROC: {pl.metrics.auroc(y_true, y_pred):.3f}")

# Token-level training with post-probe aggregation (use post.* after probe)
pipeline = pl.Pipeline([
    ("select", pre.SelectLayer(16)),
    ("probe", pl.probes.Logistic(device="cuda")),  # Token-level predictions
    ("pool", post.Pool(method="mean")),  # Aggregate scores (Scores → Scores)
])
```

**Previous Updates:**
- **Unified Activations API**:
  - `Activations.pool()`: Unified method for pooling
  - `Activations.select()`: Unified method for layer selection
  - `Activations.from_hidden_states()`: Create from HuggingFace format
- **Mask system**: Comprehensive composable mask functions
- Function-based metrics API with bootstrap confidence intervals
- Streaming support with `ActivationIterator`
- Attention probe implementation

**Future Enhancements:**
- Multi-dataset train/eval (combine datasets for training)
- TorchScript compilation for production inference
- Additional preprocessing transformers (PCA, etc.)
- Additional mask functions (token-level pattern matching, custom predicates)
- Integration utilities for external frameworks (inspect_ai, control-arena)
