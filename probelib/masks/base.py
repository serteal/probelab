"""Base classes for mask functions."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch import Tensor

from ..types import Dialogue


@dataclass
class TokenMetadata:
    """Metadata for efficient mask evaluation."""

    token_ids: Tensor  # [batch, seq_len]
    role_ids: Tensor  # [batch, seq_len] - 0=system, 1=user, 2=assistant
    message_boundaries: Tensor  # [batch, seq_len] - message index for each token
    attention_mask: Tensor  # [batch, seq_len] - 1 for real tokens, 0 for padding

    # Optional fields for text-based masks
    token_to_char: dict | None = None  # Maps token indices to char indices
    char_to_token: Callable[[int, int], int | None] | None = None  # Maps char indices to token indices
    formatted_texts: Sequence[str] | None = None  # Original formatted texts

    # Padding information
    role_ids_no_padding: Tensor | None = None  # Role IDs without padding applied
    architecture: str | None = None  # Model architecture for padding config
    special_token_ids: set[int] | None = None  # Known special token ids


class MaskFunction(ABC):
    """Composable token selector operating on :class:`TokenMetadata`.

    Masks receive both the original dialogues and precomputed metadata so they
    can implement rich selection logic without repeating tokenization work. The
    boolean operators (``&``, ``|``, ``~``) return composite masks, allowing
    callers to build complex conditions from small, well-tested pieces.
    """

    def __and__(self, other: "MaskFunction") -> "MaskFunction":
        """Logical AND with another mask."""
        from .composite import AndMask

        return AndMask(self, other)

    def __or__(self, other: "MaskFunction") -> "MaskFunction":
        """Logical OR with another mask."""
        from .composite import OrMask

        return OrMask(self, other)

    def __invert__(self) -> "MaskFunction":
        """Logical NOT."""
        from .composite import NotMask

        return NotMask(self)

    @abstractmethod
    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """
        Evaluate mask on tokenized dialogues.

        Args:
            dialogues: Original dialogues
            metadata: Token metadata for efficient evaluation

        Returns:
            Boolean tensor [batch_size, seq_len] with True for selected tokens
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
