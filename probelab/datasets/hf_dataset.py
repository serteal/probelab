"""Generic HuggingFace dataset loader with declarative configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from datasets import load_dataset

from ..types import Dialogue, DialogueDataType, Label
from .base import DialogueDataset
from .builders import (
    build_from_fields,
    build_from_messages,
    build_text_only,
    get_field,
    sample_hf_dataset,
)


@dataclass
class DatasetSpec:
    """Declarative specification for a HuggingFace dataset.

    Attributes:
        hf_path: HuggingFace dataset path (e.g., "allenai/WildChat-1M")
        hf_config: Optional dataset configuration/subset name
        hf_split: Default split to load (default: "train")

        shape: Dialogue construction method:
            - "messages": Parse from message array field
            - "fields": Build from named fields
            - "text": Single text field as dialogue
            - "custom": Use custom builder_fn

        messages_field: Field containing message array (for shape="messages")
        role_field: Field name for role in messages (default: "role")
        content_field: Field name for content in messages (default: "content")

        system_fields: Tuple of field names to try for system message
        user_fields: Tuple of field names to try for user message
        assistant_fields: Tuple of field names to try for assistant message

        text_field: Field containing text (for shape="text")
        text_as_assistant: Whether text should be assistant role (default: True)

        builder_fn: Custom dialogue builder function (for shape="custom")

        metadata_fields: Dict mapping metadata keys to tuples of field names
        label_fn: Optional function to determine label from item (default: all NEGATIVE)
        default_max_samples: Default max_samples limit (None = no limit)
    """

    # HuggingFace loading
    hf_path: str
    hf_config: str | None = None
    hf_split: str = "train"

    # Dialogue shape: "messages", "fields", "text", or "custom"
    shape: str = "fields"

    # For shape="messages"
    messages_field: str = "messages"
    role_field: str = "role"
    content_field: str = "content"
    role_fallback: str = "from"
    content_fallback: str = "value"

    # For shape="fields"
    system_fields: tuple[str, ...] = ()
    user_fields: tuple[str, ...] = ("input", "instruction", "question", "prompt")
    assistant_fields: tuple[str, ...] = ("output", "response", "answer", "solution")

    # For shape="text"
    text_field: str = "text"
    text_as_assistant: bool = True

    # For shape="custom"
    builder_fn: Callable[[dict], Dialogue] | None = None

    # Metadata fields to extract: {meta_key: (field_names...)}
    metadata_fields: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Label function (default: all NEGATIVE)
    label_fn: Callable[[dict], Label] | None = None

    # Default max_samples (None = no limit)
    default_max_samples: int | None = None


class HFDataset(DialogueDataset):
    """Generic HuggingFace dataset that uses declarative DatasetSpec.

    Subclasses only need to define:
        - base_name: str - unique identifier for the dataset
        - spec: DatasetSpec - declarative configuration

    Example:
        class WildChatDataset(HFDataset):
            base_name = "wildchat"
            spec = DatasetSpec(
                hf_path="allenai/WildChat-1M",
                shape="messages",
                messages_field="conversation",
                metadata_fields={
                    "language": ("language",),
                    "model": ("model",),
                },
            )
    """

    spec: ClassVar[DatasetSpec]

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        spec = self.spec

        # Get parameters with defaults from spec
        max_samples = kwargs.get("max_samples", spec.default_max_samples)
        split_name = kwargs.get("split", spec.hf_split)
        config = kwargs.get("config", spec.hf_config)

        # Load dataset
        if config:
            dataset = load_dataset(spec.hf_path, config)
        else:
            dataset = load_dataset(spec.hf_path)

        # Handle different return types
        if isinstance(dataset, dict):
            split = dataset[split_name]
        else:
            split = dataset

        # Apply sampling
        split = sample_hf_dataset(split, max_samples)

        # Process items
        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {k: [] for k in spec.metadata_fields}

        for item in split:
            # Build dialogue based on shape
            dialogue = self._build_dialogue(item, spec)

            if not dialogue:
                continue

            # Determine label
            label = spec.label_fn(item) if spec.label_fn else Label.NEGATIVE

            dialogues.append(dialogue)
            labels.append(label)

            # Extract metadata
            for meta_key, field_names in spec.metadata_fields.items():
                metadata[meta_key].append(get_field(item, *field_names))

        return dialogues, labels, metadata if metadata else None

    def _build_dialogue(self, item: dict, spec: DatasetSpec) -> Dialogue:
        """Build dialogue based on spec shape."""
        if spec.shape == "messages":
            messages = item.get(spec.messages_field, [])
            return build_from_messages(
                messages,
                role_field=spec.role_field,
                content_field=spec.content_field,
                role_fallback=spec.role_fallback,
                content_fallback=spec.content_fallback,
            )

        elif spec.shape == "fields":
            return build_from_fields(
                item,
                system_fields=spec.system_fields,
                user_fields=spec.user_fields,
                assistant_fields=spec.assistant_fields,
            )

        elif spec.shape == "text":
            text = get_field(item, spec.text_field)
            return build_text_only(text, spec.text_as_assistant)

        elif spec.shape == "custom":
            if spec.builder_fn:
                return spec.builder_fn(item)
            return []

        return []
