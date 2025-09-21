"""
Core types and data structures for probelib.

This module defines the fundamental data types used throughout the library,
including Messages, Dialogues, Labels, and related protocols.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """
    A single message in a dialogue.

    Attributes:
        role: The role of the message sender (system, user, assistant)
        content: The content of the message
    """

    role: Role
    content: str


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
