"""Dataset registry for easy dataset discovery and loading."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import DialogueDataset

# Registry mapping: name -> (class, category, description)
_REGISTRY: dict[str, tuple[type["DialogueDataset"], str, str]] = {}


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Handle acronyms like "AI" or "QA" by inserting underscore before them
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _get_dataset_name(cls: type) -> str:
    """Get the canonical name for a dataset class."""
    # Check if class has explicit base_name that's not the default
    base_name = getattr(cls, "base_name", None)
    if base_name and base_name != "base_dataset":
        return base_name

    # Fall back to converting class name
    name = cls.__name__
    if name.endswith("Dataset"):
        name = name[:-7]  # Remove "Dataset" suffix
    return _camel_to_snake(name)


def register(
    category: str,
    description: str = "",
) -> callable:
    """Decorator to register a dataset class.

    Args:
        category: Category for the dataset (e.g., "deception", "harmfulness")
        description: Short description of the dataset

    Example:
        @register("deception", "Deceptive chat conversations")
        class MyDataset(DialogueDataset):
            ...
    """

    def decorator(cls: type["DialogueDataset"]) -> type["DialogueDataset"]:
        name = _get_dataset_name(cls)
        _REGISTRY[name] = (cls, category, description)
        return cls

    return decorator


def load(name: str, **kwargs) -> "DialogueDataset":
    """Load a dataset by name.

    Args:
        name: Dataset name (e.g., "circuit_breakers", "dolus_chat")
        **kwargs: Additional arguments passed to the dataset constructor

    Returns:
        Instantiated dataset

    Raises:
        KeyError: If dataset name not found

    Example:
        >>> dataset = load("circuit_breakers")
        >>> dataset = load("dolus_chat", shuffle_upon_init=False)
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Dataset '{name}' not found. Available datasets: {available}"
        )

    cls, _, _ = _REGISTRY[name]
    return cls(**kwargs)


def list_datasets(category: str | None = None) -> list[str]:
    """List available dataset names.

    Args:
        category: Optional category filter (e.g., "deception", "harmfulness")

    Returns:
        Sorted list of dataset names

    Example:
        >>> list_datasets()
        ['ai_audit', 'ai_liar', 'circuit_breakers', ...]
        >>> list_datasets(category="deception")
        ['ai_audit', 'ai_liar', 'dolus_chat', ...]
    """
    if category is None:
        return sorted(_REGISTRY.keys())

    return sorted(
        name for name, (_, cat, _) in _REGISTRY.items() if cat == category
    )


def list_categories() -> list[str]:
    """List available dataset categories.

    Returns:
        Sorted list of unique category names

    Example:
        >>> list_categories()
        ['agents', 'creative', 'cybersecurity', 'deception', ...]
    """
    categories = {cat for _, (_, cat, _) in _REGISTRY.items()}
    return sorted(categories)


def info(name: str) -> dict[str, Any]:
    """Get information about a dataset.

    Args:
        name: Dataset name

    Returns:
        Dictionary with dataset info (name, category, description, class)

    Raises:
        KeyError: If dataset name not found

    Example:
        >>> info("circuit_breakers")
        {'name': 'circuit_breakers', 'category': 'harmfulness', ...}
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Dataset '{name}' not found. Available datasets: {available}"
        )

    cls, category, description = _REGISTRY[name]
    return {
        "name": name,
        "category": category,
        "description": description,
        "class": cls,
        "class_name": cls.__name__,
    }


def _init_registry() -> None:
    """Initialize the registry by importing all dataset modules.

    Each module uses @register decorators to register its datasets.
    """
    from . import (  # noqa: F401
        agents,
        creative,
        cybersecurity,
        deception,
        harmfulness,
        legal_finance,
        medical,
        mental_health,
        multilingual,
        ood,
        reasoning,
        science,
    )


# Initialize registry on module import
_init_registry()
