"""
Core types and data structures for probelab.

This module defines the fundamental data types used throughout the library,
including Messages, Dialogues, Labels, and related protocols.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class HookPoint(str, Enum):
    """Where to extract activations from the model.

    PRE_LAYERNORM: Before layer normalization (input to layernorm)
                   Earlier in computation, less processed

    POST_BLOCK: After transformer block + final layernorm
               Matches HuggingFace hidden_states semantics
               Default recommended for most use cases
    """

    PRE_LAYERNORM = "pre_layernorm"
    POST_BLOCK = "post_block"


class AggregationMethod(str, Enum):
    MEAN = "mean"
    MAX = "max"
    LAST_TOKEN = "last_token"
    EMA = "ema"
    ROLLING = "rolling"


@dataclass
class Message:
    """
    A single message in a dialogue.

    Attributes:
        role: The role of the message sender (Role enum or string: "system", "user", "assistant")
        content: The content of the message

    Examples:
        >>> # Using enum (recommended)
        >>> msg = Message(role=Role.ASSISTANT, content="Hello")

        >>> # Using string (also supported)
        >>> msg = Message(role="assistant", content="Hello")
    """

    role: Role | str
    content: str

    def __post_init__(self):
        """Convert string roles to Role enum."""
        if isinstance(self.role, str):
            try:
                self.role = Role(self.role)
            except ValueError:
                raise ValueError(
                    f"Invalid role: {self.role}. "
                    f"Must be one of: {[r.value for r in Role]}"
                )


# Type alias for a dialogue (sequence of messages)
Dialogue = list[Message]


class Label(int, Enum):
    """Labels for classification tasks.

    Inherits from int so Label values can be used directly as integers
    in mathematical operations and tensor creation.
    """

    NEGATIVE = 0
    POSITIVE = 1

    def __str__(self) -> str:
        """String representation."""
        if self == Label.NEGATIVE:
            return "negative"
        elif self == Label.POSITIVE:
            return "positive"
        else:
            raise ValueError(f"Invalid label: {self}")


# Type alias for the output of dataset loading functions
DialogueDataType = tuple[list[Dialogue], list[Label], dict[str, list[Any]] | None]
