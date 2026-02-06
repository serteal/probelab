"""
Tokenization utilities for probelab.

This module provides utilities for tokenizing dialogues and creating detection masks
using the new mask system instead of use_for_training flags.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..datasets import Dataset
from ..logger import logger
from ..masks import Mask, TokenMetadata
from ..types import Dialogue, Message
from .chat_templates import detect_template, get_template

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class Tokens:
    """Tokenized inputs ready for activation collection.

    Args:
        input_ids: Token IDs [batch, seq].
        attention_mask: Attention mask [batch, seq].
        padding_side: "left" or "right" - required for correct batch assembly.
        detection_mask: Which tokens to extract [batch, seq]. Defaults to attention_mask.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    padding_side: str
    detection_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.detection_mask is None:
            object.__setattr__(self, "detection_mask", self.attention_mask)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.input_ids.shape[0], self.input_ids.shape[1])

    def to(self, device: str | torch.device) -> "Tokens":
        return Tokens(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.padding_side,
            self.detection_mask.to(device) if self.detection_mask is not None else None,
        )

    def __getitem__(self, idx: int | slice | list[int] | torch.Tensor) -> "Tokens":
        """Slice tokens: tokens[10:20] or tokens[[0,5,10]]."""
        return Tokens(
            self.input_ids[idx],
            self.attention_mask[idx],
            self.padding_side,
            self.detection_mask[idx] if self.detection_mask is not None else None,
        )


def preprocess_dialogue(
    dialogue: Dialogue, fold_system: bool = False
) -> list[dict[str, str]]:
    """Prepare a dialogue for ``apply_chat_template`` while tracking transformations.

    - Adjacent messages with the same role are concatenated so downstream masking
      and activation collection operate on full conversational turns instead of
      fragments.
    - When ``fold_system`` is ``True`` (e.g. Gemma-style chat), system content is
      merged into the opening user message to mirror the prompt format expected by
      the tokenizer.
    - The return value is a list of ``{"role": ..., "content": ...}`` dictionaries
      understood by Hugging Face chat templates; the subsequent metadata step
      records how those processed messages align with tokens so custom masks can
      recover role/message boundaries without re-tokenizing.
    """
    processed: list[dict[str, str]] = []
    if fold_system and dialogue and dialogue[0].role == "system":
        processed.append(
            {"role": "user", "content": dialogue[0].content.strip() + "\n\n"}
        )
        dialogue = dialogue[1:]

    for message in dialogue:
        if processed and processed[-1]["role"] == message.role:
            next_content = message.content.strip()
            if next_content:
                if processed[-1]["content"]:
                    processed[-1]["content"] += " " + next_content
                else:
                    processed[-1]["content"] = next_content
        else:
            processed.append({"role": message.role, "content": message.content.strip()})

    return processed


def build_token_metadata(
    dialogues: Sequence[Dialogue],
    formatted_dialogues: Sequence[str],
    tokenizer: "PreTrainedTokenizerBase",
    tokenizer_out: dict[str, torch.Tensor],
) -> TokenMetadata:
    """Build metadata for efficient mask evaluation.

    This function creates metadata tensors that map tokens to messages and roles,
    enabling fast vectorized mask evaluation.
    """
    batch_size, seq_len = tokenizer_out["input_ids"].shape
    device = tokenizer_out["input_ids"].device

    # -1 indicates no role/message (e.g., special tokens)
    role_ids_no_padding = torch.full(
        (batch_size, seq_len), -1, dtype=torch.int8, device=device
    )
    message_boundaries = torch.full(
        (batch_size, seq_len), -1, dtype=torch.int32, device=device
    )

    # Role mapping (system is 0, user is 1, assistant is 2)
    role_to_id = {"system": 0, "user": 1, "assistant": 2}

    # Prepare BOS token IDs for optional padding-based role assignment later
    bos_token_ids = {0, 1, 2, 128000}
    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        bos_token_ids.add(tokenizer.bos_token_id)

    template_name = detect_template(tokenizer)
    template = get_template(tokenizer)
    prefix_pattern = template["prefix_pattern"]
    pad_left, pad_right = template["token_padding"]
    fold_system = template["fold_system"]

    for batch_idx, dialogue in enumerate(dialogues):
        char_idx = 0
        formatted_text = formatted_dialogues[batch_idx]

        for msg_idx, message in enumerate(dialogue):
            # For models that fold system messages into user (like Gemma),
            # system content gets merged with user, so skip explicit system messages
            if fold_system and message.role == "system":
                # The content will be part of the user message in the formatted text
                continue

            # Find the start of the message content
            match = re.match(prefix_pattern, formatted_text[char_idx:])
            if match is None:
                logger.warning(f"Could not match prefix pattern at position {char_idx}")
                continue

            start_char_idx = char_idx + match.end()
            end_char_idx = start_char_idx + len(message.content.strip())

            # Find corresponding token indices. ``char_to_token`` returns the index of the
            # token covering a particular character position. The exclusive end of the
            # slice therefore needs the token index of the *last* content character plus
            # one; otherwise we incorrectly include the remainder of the sequence when the
            # end character falls exactly on a token boundary.
            start_tok_idx = tokenizer_out.char_to_token(batch_idx, start_char_idx)

            end_tok_inclusive = None
            if len(message.content.strip()) > 0:
                end_tok_inclusive = tokenizer_out.char_to_token(
                    batch_idx, max(start_char_idx, end_char_idx - 1)
                )

            if start_tok_idx is not None:
                if end_tok_inclusive is None:
                    # char_to_token can return ``None`` when the message is empty or when
                    # we're at the very end of the decoded sequence. Fall back to the
                    # start token so we at least cover the first character instead of
                    # spilling to the rest of the sequence.
                    end_tok_inclusive = start_tok_idx

                exclusive_end = min(seq_len, end_tok_inclusive + 1)

                role_id = role_to_id[message.role]
                role_ids_no_padding[batch_idx, start_tok_idx:exclusive_end] = role_id
                message_boundaries[batch_idx, start_tok_idx:exclusive_end] = msg_idx

            char_idx = end_char_idx

    role_ids_with_padding = role_ids_no_padding.clone()

    if pad_left > 0 or pad_right > 0:
        for batch_idx in range(batch_size):
            for role_id in [0, 1, 2]:  # system, user, assistant
                role_mask = role_ids_no_padding[batch_idx] == role_id

                if not role_mask.any():
                    continue

                padded_mask = torch.cat(
                    [
                        torch.tensor([False], device=device),
                        role_mask,
                        torch.tensor([False], device=device),
                    ]
                )

                diff = padded_mask[1:].int() - padded_mask[:-1].int()
                starts = torch.where(diff == 1)[0]
                ends = torch.where(diff == -1)[0]

                for start, end in zip(starts, ends):
                    padded_start = max(0, start - pad_left)
                    padded_end = min(seq_len, end + pad_right)
                    role_ids_with_padding[batch_idx, padded_start:padded_end] = role_id

    # Mark BOS tokens as system only in the padded view (include_padding=True)
    for batch_idx in range(batch_size):
        first_token_id = tokenizer_out["input_ids"][batch_idx, 0].item()
        if first_token_id in bos_token_ids:
            role_ids_with_padding[batch_idx, 0] = role_to_id["system"]

    special_token_ids: set[int] | None = None
    if hasattr(tokenizer, "all_special_ids") and tokenizer.all_special_ids is not None:
        try:
            special_token_ids = {
                int(token_id)
                for token_id in tokenizer.all_special_ids
                if token_id is not None
            }
        except TypeError:
            # Fall back if tokenizer reports non-iterable
            pass

    return TokenMetadata(
        token_ids=tokenizer_out["input_ids"],
        role_ids=role_ids_with_padding,  # Use padded version by default
        message_boundaries=message_boundaries,
        attention_mask=tokenizer_out["attention_mask"],
        char_to_token=tokenizer_out.char_to_token
        if hasattr(tokenizer_out, "char_to_token")
        else None,
        token_to_char=None,  # Not available in tokenizer output
        formatted_texts=formatted_dialogues,
        role_ids_no_padding=role_ids_no_padding,  # Store unpadded version
        architecture=template_name,  # Store template name
        special_token_ids=special_token_ids,
    )


def tokenize_dialogues(
    tokenizer: "PreTrainedTokenizerBase",
    dialogues: Sequence[Dialogue],
    mask: Mask,
    device: torch.device | str = "cpu",
    add_generation_prompt: bool = False,
    template_kwargs: dict[str, Any] | None = None,
    **tokenize_kwargs: Any,
) -> Tokens:
    """Tokenize dialogues with explicit mask for detection control.

    Args:
        tokenizer: HuggingFace tokenizer bound to the model.
        dialogues: Sequence of dialogues to tokenize.
        mask: Mask function determining which tokens to detect. Required.
            Use ``masks.all()`` to detect all non-padding tokens.
        device: Device to place tensors on.
        add_generation_prompt: Whether to append the model's generation prompt.
        template_kwargs: Extra keyword arguments forwarded to
            ``tokenizer.apply_chat_template()`` (e.g. ``{"enable_thinking": True}``
            for Qwen3).
        **tokenize_kwargs: Additional tokenizer arguments forwarded verbatim.

    Returns:
        Tokens object with input_ids, attention_mask, detection_mask.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert dialogues to format expected by tokenizer
    fold_system = get_template(tokenizer)["fold_system"]
    dialogue_dicts = [
        preprocess_dialogue(dialogue, fold_system) for dialogue in dialogues
    ]

    # Build a processed Dialogue (Message objects) aligned with formatted text
    processed_dialogues: list[list[Message]] = [
        [Message(role=m["role"], content=m["content"]) for m in d]
        for d in dialogue_dicts
    ]

    # Apply chat template if available, otherwise use simple formatting
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        apply_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if template_kwargs:
            apply_kwargs.update(template_kwargs)
        formatted_dialogues = tokenizer.apply_chat_template(
            dialogue_dicts,
            **apply_kwargs,
        )
    else:
        # Fallback for models without chat templates (e.g., LLAMA-2)
        # For simple single-turn dialogues, just use the content directly
        formatted_dialogues = []
        for dialogue_dict in dialogue_dicts:
            if len(dialogue_dict) == 1 and dialogue_dict[0]["role"] == "user":
                # Single user message - just use the content
                formatted_dialogues.append(dialogue_dict[0]["content"])
            else:
                # Multi-turn dialogue - format as simple text
                formatted = ""
                for msg in dialogue_dict:
                    if msg["role"] == "system":
                        formatted += f"System: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        formatted += f"User: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        formatted += f"Assistant: {msg['content']}\n\n"
                formatted_dialogues.append(formatted.strip())

    default_tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
    }
    default_tokenize_kwargs.update(tokenize_kwargs)

    # Tokenize
    token_dict = tokenizer(formatted_dialogues, **default_tokenize_kwargs)  # type: ignore

    # Move to device
    for k, v in token_dict.items():
        if isinstance(v, torch.Tensor):
            token_dict[k] = v.to(device)
        elif isinstance(v, list):
            token_dict[k] = torch.tensor(v, device=device)

    if "attention_mask" not in token_dict:
        raise ValueError("Tokenizer output must include attention mask")

    # Build metadata for mask evaluation
    metadata = build_token_metadata(
        processed_dialogues, formatted_dialogues, tokenizer, token_dict
    )

    # Evaluate mask and create detection_mask
    detection_mask = mask(dialogues, metadata)

    return Tokens(
        input_ids=token_dict["input_ids"],
        attention_mask=token_dict["attention_mask"],
        padding_side=getattr(tokenizer, "padding_side", "right"),
        detection_mask=detection_mask,
    )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: "PreTrainedTokenizerBase",
    mask: Mask,
    device: torch.device | str = "cpu",
    template_kwargs: dict[str, Any] | None = None,
    **tokenize_kwargs: Any,
) -> Tokens:
    """Tokenize a dataset with explicit mask for detection control.

    Args:
        dataset: Dataset to tokenize.
        tokenizer: HuggingFace tokenizer aligned with the model.
        mask: Mask function determining which tokens to detect. Required.
            Use ``masks.all()`` to detect all non-padding tokens.
        device: Device to place tensors on.
        template_kwargs: Extra keyword arguments forwarded to
            ``tokenizer.apply_chat_template()`` (e.g. ``{"enable_thinking": True}``
            for Qwen3).
        **tokenize_kwargs: Additional tokenizer arguments.

    Returns:
        Tokens object with input_ids, attention_mask, detection_mask.
    """
    return tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dataset.dialogues,
        mask=mask,
        device=device,
        template_kwargs=template_kwargs,
        **tokenize_kwargs,
    )
