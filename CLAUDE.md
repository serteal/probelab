# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Information

- **Name**: probelib
- **Version**: 0.1.0
- **Author**: Alex Serrano
- **Python**: >=3.11
- **Build System**: hatchling (modern Python packaging)
- **License**: Not specified (recommend adding)

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
from probelib import Message, Dialogue, Label

# Activation handling
from probelib import HookedModel, Activations, collect_activations

# Probes
from probelib.probes import BaseProbe, Logistic, MLP, Attention

# High-level workflows
from probelib.scripts import train_probes, evaluate_probes

# Datasets
from probelib.datasets import DialogueDataset, CircuitBreakersDataset, DolusChatDataset

# Metrics
from probelib.metrics import auroc, recall_at_fpr, get_metric_by_name

# Visualization
from probelib.visualization import print_metrics
```

## Commands

### Development Setup

```bash
# Clone and install in development mode
git clone <repo_url>
cd probelib
uv sync --dev

# Verify installation
uv run python -c "import probelib; print(probelib.__version__)"
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
uv run pytest tests/processing/test_aggregation.py -v
uv run pytest tests/probes/test_linear.py -v
uv run pytest tests/probes/test_mlp.py -v

# Run tests for specific model
uv run pytest tests/ -k "llama" -v
uv run pytest tests/ -k "gemma" -v

# Run tests by marker
uv run pytest tests/ -m "not slow"      # Exclude slow tests
uv run pytest tests/ -m "requires_gpu"  # GPU tests only

# Run with coverage
uv run pytest tests/ --cov=probelib --cov-report=html --cov-report=term
```

### Code Quality

```bash
# Run linting (configured to ignore F722 for jaxtyping)
uv run ruff check src/
uv run ruff format src/

# Type checking (if mypy is added)
# uv run mypy src/probelib
```

### Build and Distribution

```bash
# Build the package
uv build

# The build creates:
# - dist/probelib-0.1.0-py3-none-any.whl
# - dist/probelib-0.1.0.tar.gz

# Install from wheel
pip install dist/probelib-0.1.0-py3-none-any.whl
```

### Development Dependencies

The `dev` dependency group includes:

- `huggingface-hub`: For downloading models and datasets from HF Hub
- `ipykernel`: Jupyter notebook kernel for interactive development
- `ipywidgets`: Interactive widgets for notebooks
- `python-dotenv`: Environment variable management (.env files)

### Running Examples

```bash
# Start Jupyter for interactive development
uv run jupyter notebook

# Run example scripts
uv run python examples/train_streaming_simple.py  # Simple streaming training
uv run python examples/perf_example.py            # Performance benchmarking
uv run python examples/dolus_chat.py              # Interactive deception chat demo
```

## Architecture Overview

probelib is a library for training classifiers (probes) on Large Language Model (LLM) activations to understand what information is encoded in different model layers. It's designed for interpretability research, particularly for understanding how LLMs represent concepts internally.

### Core Concepts

1. **Probes**: Classifiers trained on LLM activations to detect specific properties
2. **Dialogues**: Structured conversations with detection flags for training
3. **Activations**: Hidden states extracted from LLM layers during forward passes
4. **Detection Masks**: Binary masks indicating which tokens to use for training

### Module Structure

```
probelib/
├── __init__.py          # Public API exports
├── types.py             # Core type definitions
├── datasets/            # Dataset handling
│   ├── base.py         # DialogueDataset base class
│   ├── deception.py    # Deception detection datasets
│   ├── harmfulness.py  # Harmfulness detection datasets
│   ├── language.py     # Language detection datasets
│   └── ood.py          # Out-of-distribution datasets
├── models/              # Model interfaces
│   ├── architectures.py # Model-specific configurations
│   └── hooks.py        # PyTorch hook management
├── processing/          # Data processing
│   ├── activations.py  # Activation extraction and containers
│   └── tokenization.py # Dialogue tokenization
├── probes/              # Probe implementations
│   ├── base.py         # BaseProbe abstract class
│   ├── logistic.py     # Logistic regression probes
│   ├── mlp.py          # MLP probe
│   └── attention.py    # Attention-based probe
├── scripts/             # High-level workflows
│   └── workflows.py    # train_probes, evaluate_probes
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
     InsiderTradingDataset, plus Cadenza datasets (ConvincingGame, HarmPressure,
     InstructDishonesty, MaskDataset, InsiderTrading)
   - **Harmfulness datasets**: CircuitBreakersDataset, BenignInstructionsDataset,
     WildJailbreakDataset, WildGuardMixDataset, XSTestResponseDataset, CoconotDataset,
     ToxicChatDataset, ClearHarmLlama3Dataset, ClearHarmMistralSmallDataset
   - **Language datasets**: Support for 20+ languages including Arabic, Chinese, French,
     German, Hindi, Korean, Spanish, etc.
   - **OOD datasets**: AlpacaDataset, LmsysChatDataset, MATHInstructionDataset,
     UltraChatDataset, and more

3. **Models (`models/`)**: LLM interfaces

   - `HookedModel`: Context manager for activation extraction
   - Architecture-specific handlers (LLaMA, Gemma)
   - Automatic layer detection and hook management
   - Memory-efficient activation collection

4. **Processing (`processing/`)**: Data transformation pipeline

   - `collect_activations`: Extract hidden states from models (main API)
   - `Activations`: Container class with methods:
     - `aggregate()`: Aggregate over sequence (mean, max, last_token)
     - `to_token_level()`: Extract token-level features
     - `filter_layers()`: Filter to specific layers
     - `to()`: Device/dtype conversion
   - `ActivationIterator`: Memory-efficient streaming wrapper
   - `tokenize_dialogues`: Convert dialogues to model inputs
   - `create_detection_mask`: Mark tokens for probe training

5. **Probes (`probes/`)**: Classifier implementations

   - **BaseProbe**: Abstract base with unified aggregation handling
   - **Logistic**: L2-regularized logistic regression
   - **SklearnLogistic**: Scikit-learn based variant
   - **MLP**: Multi-layer perceptron with dropout
   - **Attention**: Attention-weighted classification
   - Aggregation modes:
     - `sequence_aggregation`: Aggregate BEFORE training (classic)
     - `score_aggregation`: Train on tokens, aggregate AFTER prediction
   - All support streaming via `partial_fit()`

6. **Scripts (`scripts/workflows.py`)**: High-level workflows

   - `train_probes`: Unified training interface
   - `evaluate_probes`: Comprehensive evaluation with metrics
   - Automatic streaming detection for large datasets
   - Multi-probe training with shared activations

7. **Metrics (`metrics.py`)**: Function-based metrics API

   - Core metrics: `auroc`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`
   - Special metrics: `recall_at_fpr`, `partial_auroc`, `fpr_at_threshold`
   - Bootstrap confidence intervals via `@with_bootstrap()` decorator
   - String-based metric lookup: `get_metric_by_name("auroc")`
   - Parameterized metrics: `"recall@5"` (5% FPR), `"percentile95"`

8. **Visualization (`visualization.py`)**: Result visualization
   - ROC and precision-recall curves
   - Recall comparison bar charts
   - Detection mask visualization
   - Modern plotting theme with accessibility

### Key Design Patterns

1. **Detection-First Design**

   - The `detect` flag on messages controls training data
   - Allows fine-grained control over which parts of conversations to use
   - Essential for avoiding label leakage in dialogue contexts

2. **Memory Efficiency**

   - Streaming activation collection via `ActivationIterator`
   - Automatic streaming detection based on dataset size
   - Dynamic batching based on sequence lengths
   - Tensor views for efficient batch processing
   - Optional model pruning (keep only needed layers)
   - Activation caching with deterministic hashing
   - `partial_fit()` support for incremental training

3. **Unified Interfaces**

   - Single `train_probes` function for all probe types
   - Consistent configuration via `ProbeConfig`
   - Automatic format detection and conversion
   - Works with both single probes and probe dictionaries

4. **Extensibility**
   - Abstract base classes for datasets and probes
   - Plugin architecture for new model support
   - Configurable aggregation and transformation
   - Custom metrics via string identifiers

### Common Workflows

1. **Basic Probe Training**

   ```python
   import probelib as pl
   from probelib.scripts import train_probes

   # Load data
   dataset = pl.datasets.CircuitBreakersDataset()

   # Create probe with aggregation
   probe = pl.probes.Logistic(
       layer=12,
       sequence_aggregation="mean"  # Or score_aggregation for token-level
   )

   # Train
   train_probes(probe, model, tokenizer, dataset, dataset.labels)
   ```

2. **Multi-Layer Analysis**

   ```python
   # Train probes on multiple layers
   probes = {
       f"layer_{i}": pl.probes.Logistic(layer=i, sequence_aggregation="mean")
       for i in range(model.config.num_hidden_layers)
   }
   pl.scripts.train_probes(probes, model, tokenizer, dataset, dataset.labels)
   ```

3. **Streaming for Large Datasets**
   ```python
   # Streaming mode for memory efficiency
   pl.scripts.train_probes(
       probe, model, tokenizer, large_dataset, labels,
       streaming=True,  # Returns ActivationIterator
       batch_size=8
   )
   ```

4. **Evaluation with Custom Metrics**
   ```python
   from probelib.scripts import evaluate_probes
   import functools

   # Evaluate with custom metrics
   predictions, metrics = evaluate_probes(
       probe, model, tokenizer, test_data, test_labels,
       metrics=[
           "auroc",
           "balanced_accuracy",
           functools.partial(pl.metrics.recall_at_fpr, fpr=0.01)
       ]
   )
   ```

### Adding New Features

**New Probe Types:**

1. Create class inheriting from `BaseProbe` in `probes/`
2. Implement required methods: `fit`, `predict`, `partial_fit` (for streaming)
3. Handle aggregation via `_prepare_features()` and `_aggregate_scores()`
4. Add comprehensive tests in `tests/probes/`
5. Update documentation

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
   - Test with show_detection_mask_in_html

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

```bash
# Logging level
PROBELIB_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Cache directory
PROBELIB_CACHE_DIR=/path/to/cache

# Default device
PROBELIB_DEVICE=cuda:0  # or cpu

# Disable progress bars
PROBELIB_DISABLE_PROGRESS=1
```

### Pytest Configuration

Located in `pyproject.toml`:

- Test discovery: `tests/` directory
- Naming convention: `test_*.py` files
- Custom markers: `slow`, `requires_gpu`
- Coverage targets: 80%+ recommended

### Recent Updates & Future Enhancements

**Recently Added:**
- Function-based metrics API with bootstrap confidence intervals
- Unified aggregation API (sequence vs score aggregation)
- Streaming support with `ActivationIterator`
- Multi-probe parallel training with shared activations
- Attention probe implementation
- Extensive Cadenza dataset collection

**Future Enhancements:**
- Multi-layer probe support (currently single layer only)
- Multi-dataset train/eval (combine datasets for training)
- Benchmark evaluation functions (standardized evaluation protocols)
- TorchScript compilation for production inference
