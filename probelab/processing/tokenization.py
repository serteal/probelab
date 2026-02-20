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
import torch.nn.functional as F

from ..datasets import Dataset
from ..logger import logger
from ..masks import Mask, TokenMetadata
from ..types import Dialogue, Message
from .chat_templates import detect_template, get_template

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class Tokens:
    """Tokenized inputs in flat+offsets layout (no padding stored).

    Args:
        input_ids: Flat token IDs ``[total_tokens]``.
        offsets: Cumulative token counts ``[N+1]`` int64.
            Sample *i* spans ``input_ids[offsets[i]:offsets[i+1]]``.
        detection_mask: Per-token detection flag ``[total_tokens]`` bool.
        pad_token_id: Padding token ID needed by backends.
        padding_side: ``"left"`` or ``"right"`` for batch materialisation.
        formatted_texts: Chat-templated strings per sample.
    """

    input_ids: torch.Tensor
    offsets: torch.Tensor
    detection_mask: torch.Tensor
    pad_token_id: int
    padding_side: str
    formatted_texts: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return self.offsets.shape[0] - 1

    @property
    def lengths(self) -> torch.Tensor:
        """Per-sample token counts ``[N]``."""
        return self.offsets[1:] - self.offsets[:-1]

    @property
    def seq_len(self) -> int:
        """Maximum sequence length across all samples."""
        lengths = self.lengths
        return int(lengths.max().item()) if lengths.numel() > 0 else 0

    @property
    def total_tokens(self) -> int:
        return self.input_ids.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """``(batch, max_seq)`` for API compat."""
        return (len(self), self.seq_len)

    def to(self, device: str | torch.device) -> "Tokens":
        return Tokens(
            input_ids=self.input_ids.to(device),
            offsets=self.offsets.to(device),
            detection_mask=self.detection_mask.to(device),
            pad_token_id=self.pad_token_id,
            padding_side=self.padding_side,
            formatted_texts=self.formatted_texts,
        )

    def __getitem__(self, idx: int | slice | list[int] | torch.Tensor) -> "Tokens":
        """Slice tokens: ``tokens[10:20]`` or ``tokens[[0,5,10]]``."""
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
        elif isinstance(idx, torch.Tensor):
            indices = idx.tolist()
        else:
            indices = list(idx)

        chunks_ids: list[torch.Tensor] = []
        chunks_det: list[torch.Tensor] = []
        new_offsets = torch.zeros(len(indices) + 1, dtype=torch.int64)
        for j, i in enumerate(indices):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            chunks_ids.append(self.input_ids[s:e])
            chunks_det.append(self.detection_mask[s:e])
            new_offsets[j + 1] = new_offsets[j] + (e - s)

        texts = None
        if self.formatted_texts is not None:
            texts = tuple(self.formatted_texts[i] for i in indices)

        return Tokens(
            input_ids=torch.cat(chunks_ids) if chunks_ids else self.input_ids.new_empty(0),
            offsets=new_offsets,
            detection_mask=torch.cat(chunks_det) if chunks_det else torch.empty(0, dtype=torch.bool),
            pad_token_id=self.pad_token_id,
            padding_side=self.padding_side,
            formatted_texts=texts,
        )

    def pad_batch(
        self,
        indices: list[int],
        padding_side: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Materialise padded ``{input_ids, attention_mask, detection_mask}`` for a subset.

        Pads to the LOCAL max sequence length of the selected samples.

        Args:
            indices: Sample indices to extract.
            padding_side: Override padding side (defaults to ``self.padding_side``).

        Returns:
            Dict with ``input_ids [B, local_max]``, ``attention_mask [B, local_max]``,
            ``detection_mask [B, local_max]``.
        """
        side = padding_side or self.padding_side
        sub_batch = len(indices)

        # Compute local max
        local_max = 0
        for i in indices:
            length = int(self.offsets[i + 1]) - int(self.offsets[i])
            local_max = max(local_max, length)

        input_ids = torch.full(
            (sub_batch, local_max), self.pad_token_id,
            dtype=self.input_ids.dtype, device=self.input_ids.device,
        )
        attention_mask = torch.zeros(
            sub_batch, local_max, dtype=torch.long, device=self.input_ids.device,
        )
        detection_mask = torch.zeros(
            sub_batch, local_max, dtype=self.detection_mask.dtype,
            device=self.detection_mask.device,
        )

        for j, i in enumerate(indices):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            length = e - s
            if length == 0:
                continue
            if side == "right":
                input_ids[j, :length] = self.input_ids[s:e]
                attention_mask[j, :length] = 1
                detection_mask[j, :length] = self.detection_mask[s:e]
            else:
                input_ids[j, local_max - length :] = self.input_ids[s:e]
                attention_mask[j, local_max - length :] = 1
                detection_mask[j, local_max - length :] = self.detection_mask[s:e]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "detection_mask": detection_mask,
        }


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


def _char_to_token_from_offsets(
    offsets: torch.Tensor, char_pos: int
) -> int | None:
    """Find the token index covering ``char_pos`` using an offset tensor.

    Args:
        offsets: [seq, 2] tensor of (start_char, end_char) per token.
        char_pos: Character position to look up.

    Returns:
        Token index or ``None`` if no token covers the position.
    """
    mask = (offsets[:, 0] <= char_pos) & (offsets[:, 1] > char_pos)
    indices = mask.nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        return None
    return int(indices[0].item())


def _expand_role_padding(
    role_ids_no_padding: torch.Tensor,
    pad_left: int,
    pad_right: int,
) -> torch.Tensor:
    """Expand role regions by ``pad_left``/``pad_right`` tokens using max-pool.

    For each role, the set of assigned token positions is dilated so that
    ``pad_left`` extra tokens to the left and ``pad_right`` extra tokens to the
    right of every contiguous region also receive that role id.  The operation
    is equivalent to morphological dilation and is implemented with a single
    ``F.max_pool1d`` call per role (fully vectorised over the batch).

    Args:
        role_ids_no_padding: [batch, seq] int8 tensor with role ids (0/1/2) and
            -1 for unassigned positions.
        pad_left: Number of tokens to expand to the left.
        pad_right: Number of tokens to expand to the right.

    Returns:
        [batch, seq] int8 tensor with expanded role assignments.
    """
    role_ids_with_padding = role_ids_no_padding.clone()
    kernel_size = pad_left + pad_right + 1

    for role_id in [0, 1, 2]:  # system, user, assistant
        role_mask = (role_ids_no_padding == role_id).float().unsqueeze(1)  # [B,1,S]
        # Pad so the max-pool window is centred correctly for asymmetric
        # expansion: pad_right zeros on the left, pad_left zeros on the right.
        padded = F.pad(role_mask, (pad_right, pad_left), value=0.0)
        dilated = F.max_pool1d(padded, kernel_size=kernel_size, stride=1)
        dilated = dilated.squeeze(1).bool()  # [B, S]
        role_ids_with_padding[dilated] = role_id

    return role_ids_with_padding


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

    # Use offset_mapping for fast vectorised char->token lookup when available.
    offset_mapping: torch.Tensor | None = None
    if "offset_mapping" in tokenizer_out:
        raw = tokenizer_out["offset_mapping"]
        if isinstance(raw, torch.Tensor):
            offset_mapping = raw
        else:
            try:
                offset_mapping = torch.tensor(raw)
            except (ValueError, TypeError):
                pass  # fall back to char_to_token

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

            # Find corresponding token indices.
            if offset_mapping is not None:
                offsets = offset_mapping[batch_idx]  # [seq, 2]
                start_tok_idx = _char_to_token_from_offsets(offsets, start_char_idx)
                end_tok_inclusive = None
                if len(message.content.strip()) > 0:
                    end_tok_inclusive = _char_to_token_from_offsets(
                        offsets, max(start_char_idx, end_char_idx - 1)
                    )
            else:
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

    # --- Padding expansion (vectorised via max-pool) ---
    if pad_left > 0 or pad_right > 0:
        role_ids_with_padding = _expand_role_padding(
            role_ids_no_padding, pad_left, pad_right
        )
    else:
        role_ids_with_padding = role_ids_no_padding.clone()

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


def _format_dialogues(
    tokenizer: "PreTrainedTokenizerBase",
    dialogue_dicts: list[list[dict[str, str]]],
    add_generation_prompt: bool,
    template_kwargs: dict[str, Any] | None,
) -> list[str]:
    """Apply chat template or fallback formatting to produce text strings."""
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        apply_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if template_kwargs:
            apply_kwargs.update(template_kwargs)
        return tokenizer.apply_chat_template(
            dialogue_dicts,
            **apply_kwargs,
        )
    else:
        formatted_dialogues: list[str] = []
        for dialogue_dict in dialogue_dicts:
            if len(dialogue_dict) == 1 and dialogue_dict[0]["role"] == "user":
                formatted_dialogues.append(dialogue_dict[0]["content"])
            else:
                formatted = ""
                for msg in dialogue_dict:
                    if msg["role"] == "system":
                        formatted += f"System: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        formatted += f"User: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        formatted += f"Assistant: {msg['content']}\n\n"
                formatted_dialogues.append(formatted.strip())
        return formatted_dialogues


def tokenize_dialogues(
    tokenizer: "PreTrainedTokenizerBase",
    dialogues: Sequence[Dialogue],
    mask: Mask,
    device: torch.device | str = "cpu",
    add_generation_prompt: bool = False,
    template_kwargs: dict[str, Any] | None = None,
    chunk_size: int = 1024,
    **tokenize_kwargs: Any,
) -> Tokens:
    """Tokenize dialogues with explicit mask for detection control.

    Processes dialogues in chunks of ``chunk_size`` to avoid O(N * max_seq) padding
    overhead when the dataset has highly variable sequence lengths.

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
        chunk_size: Number of dialogues to tokenize per chunk (default 1024).
        **tokenize_kwargs: Additional tokenizer arguments forwarded verbatim.

    Returns:
        Tokens object with flat input_ids, offsets, and detection_mask.
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

    # Format all dialogues (string ops, cheap)
    all_formatted = _format_dialogues(
        tokenizer, dialogue_dicts, add_generation_prompt, template_kwargs,
    )

    default_tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
    }
    # Note: return_offsets_mapping is intentionally NOT requested by default.
    # For fast tokenizers, char_to_token() on the BatchEncoding is ~3x faster
    # than constructing offset tensors, especially for variable-length batches.
    default_tokenize_kwargs.update(tokenize_kwargs)

    # Accumulate flat tokens across chunks
    all_ids: list[torch.Tensor] = []
    all_det: list[torch.Tensor] = []
    sample_lengths: list[int] = []

    n = len(dialogues)
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_formatted = all_formatted[chunk_start:chunk_end]
        chunk_processed = processed_dialogues[chunk_start:chunk_end]
        chunk_dialogues = dialogues[chunk_start:chunk_end]

        # Tokenize chunk (pads to chunk-local max only)
        token_dict = tokenizer(chunk_formatted, **default_tokenize_kwargs)  # type: ignore

        # Move to device
        for k, v in token_dict.items():
            if isinstance(v, torch.Tensor):
                token_dict[k] = v.to(device)
            elif isinstance(v, list):
                token_dict[k] = torch.tensor(v, device=device)

        if "attention_mask" not in token_dict:
            raise ValueError("Tokenizer output must include attention mask")

        # Build metadata for mask evaluation (operates on chunk-sized padded tensors)
        metadata = build_token_metadata(
            chunk_processed, chunk_formatted, tokenizer, token_dict
        )

        # Evaluate mask
        detection_mask_chunk = mask(chunk_dialogues, metadata)

        # Strip padding per sample, append to flat lists
        attn = token_dict["attention_mask"]
        ids = token_dict["input_ids"]
        chunk_batch = ids.shape[0]
        padding_side = getattr(tokenizer, "padding_side", "right")

        for i in range(chunk_batch):
            valid = int(attn[i].sum().item())
            if valid <= 0:
                sample_lengths.append(0)
                continue
            if padding_side == "right":
                all_ids.append(ids[i, :valid])
                all_det.append(detection_mask_chunk[i, :valid])
            else:
                all_ids.append(ids[i, -valid:])
                all_det.append(detection_mask_chunk[i, -valid:])
            sample_lengths.append(valid)

    # Build flat tensors and offsets
    if all_ids:
        flat_ids = torch.cat(all_ids, dim=0)
        flat_det = torch.cat(all_det, dim=0)
    else:
        flat_ids = torch.empty(0, dtype=torch.long, device=device)
        flat_det = torch.empty(0, dtype=torch.bool, device=device)

    offsets = torch.zeros(n + 1, dtype=torch.int64)
    for i, length in enumerate(sample_lengths):
        offsets[i + 1] = offsets[i] + length

    return Tokens(
        input_ids=flat_ids,
        offsets=offsets,
        detection_mask=flat_det.bool(),
        pad_token_id=int(tokenizer.pad_token_id),
        padding_side=getattr(tokenizer, "padding_side", "right"),
        formatted_texts=tuple(all_formatted),
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
        Tokens object with flat input_ids, offsets, and detection_mask.
    """
    return tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dataset.dialogues,
        mask=mask,
        device=device,
        template_kwargs=template_kwargs,
        **tokenize_kwargs,
    )
