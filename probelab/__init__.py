"""
probelab: A library for training classifiers on LLM activations.

Primary API:
    Pipeline, Activations, collect_activations, Label, Context

Submodule access:
    probelab.preprocessing  - SelectLayer, Pool, Normalize
    probelab.probes         - Logistic, MLP, Attention
    probelab.masks          - assistant(), user(), contains()
    probelab.datasets       - CircuitBreakersDataset, REPEDataset, etc.
    probelab.metrics        - auroc, recall_at_fpr, etc.
    probelab.scripts        - train_pipelines, evaluate_pipelines

Explicit imports for advanced use:
    from probelab.types import Message, Dialogue, Role, HookPoint
    from probelab.models import HookedModel
    from probelab.visualization import print_metrics, visualize_mask
    from probelab.config import get_config, set_defaults
    from probelab.profiling import ProbelabCounters, profile_section
"""

# Submodules (for pl.submodule.X access)
from . import datasets, masks, metrics, preprocessing, probes, scripts

# Primary API
from .config import Context
from .logger import logger
from .pipeline import Pipeline
from .processing import Activations, collect_activations
from .types import Label

__version__ = "0.1.0"

__all__ = [
    # Primary API (what 90% of users need)
    "Pipeline",
    "Activations",
    "collect_activations",
    "Label",
    "Context",
    "logger",
    # Submodule access
    "preprocessing",
    "probes",
    "masks",
    "datasets",
    "metrics",
    "scripts",
]
