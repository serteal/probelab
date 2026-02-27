"""Dataset registry - simple dict mapping names to loader functions."""

from enum import Enum

from .base import Dataset, LoaderFn


class Topic(Enum):
    """Dataset topic categories."""
    DECEPTION = "deception"
    HARMFULNESS = "harmfulness"
    OOD = "ood"
    MULTILINGUAL = "multilingual"
    MEDICAL = "medical"
    MENTAL_HEALTH = "mental_health"
    CYBERSECURITY = "cybersecurity"
    AGENTS = "agents"
    REASONING = "reasoning"
    CREATIVE = "creative"
    LEGAL_FINANCE = "legal_finance"
    SCIENCE = "science"
    HALLUCINATION = "hallucination"
    SPARSE_PROBING = "sparse_probing"


# Registry: name -> (loader_fn, category, description)
REGISTRY: dict[str, tuple[LoaderFn, str, str]] = {}
_REGISTRY_INITIALIZED = False


def _register_dataset(name: str, topic: Topic, description: str = "") -> callable:
    """Register a loader function."""
    def decorator(fn: LoaderFn) -> LoaderFn:
        REGISTRY[name] = (fn, topic.value, description)
        return fn
    return decorator


def load(name: str, **kwargs) -> Dataset:
    """Load a dataset by name."""
    _ensure_registry_initialized()
    if name not in REGISTRY:
        raise KeyError(f"Dataset '{name}' not found. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[name][0](**kwargs)


def list_datasets(category: str | None = None) -> list[str]:
    """List available dataset names, optionally filtered by category."""
    _ensure_registry_initialized()
    if category is None:
        return sorted(REGISTRY.keys())
    return sorted(n for n, (_, c, _) in REGISTRY.items() if c == category)


def list_categories() -> list[str]:
    """List available dataset categories."""
    _ensure_registry_initialized()
    return sorted({c for _, (_, c, _) in REGISTRY.items()})


def info(name: str) -> dict:
    """Get information about a dataset."""
    _ensure_registry_initialized()
    if name not in REGISTRY:
        raise KeyError(f"Dataset '{name}' not found. Available: {sorted(REGISTRY.keys())}")
    fn, category, description = REGISTRY[name]
    return {"name": name, "category": category, "description": description, "loader": fn}


def _init_registry() -> None:
    """Initialize registry by importing all dataset modules.

    Decorators on each loader function register them on import.
    """
    from . import (  # noqa: F401
        agents,
        creative,
        cybersecurity,
        deception,
        hallucination,
        harmfulness,
        legal_finance,
        medical,
        mental_health,
        multi_turn,
        multilingual,
        ood,
        reasoning,
        science,
        sparse_probing,
        spam,
        toxicity,
    )


def _ensure_registry_initialized() -> None:
    """Import and register dataset loaders on first dataset API use."""
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return
    _init_registry()
    _REGISTRY_INITIALIZED = True
