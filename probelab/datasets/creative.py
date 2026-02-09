"""Creative writing and roleplay datasets."""

from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .builders import build_from_messages
from .registry import Topic, _register_dataset


@_register_dataset("synthetic_persona_chat", Topic.CREATIVE, "Synthetic persona chat")
def synthetic_persona_chat() -> Dataset:
    """Synthetic Persona Chat 20K+ from Google."""
    data = load_dataset("google/Synthetic-Persona-Chat")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"user1_personas": [], "user2_personas": []}

    for item in data:
        conversation = item.get("Best Generated Conversation", "")
        if not conversation:
            continue

        dialogue = []
        p1 = item.get("user 1 personas", "")
        p2 = item.get("user 2 personas", "")
        if p1 or p2:
            dialogue.append(Message("system", f"User 1 personas: {p1}\nUser 2 personas: {p2}"))

        for part in conversation.split("User "):
            part = part.strip()
            if part.startswith("1:"):
                dialogue.append(Message("user", part[2:].strip()))
            elif part.startswith("2:"):
                dialogue.append(Message("assistant", part[2:].strip()))

        if len(dialogue) > 1:  # Has more than just system message
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)
            metadata["user1_personas"].append(p1)
            metadata["user2_personas"].append(p2)

    return Dataset(dialogues, labels, "synthetic_persona_chat", metadata).shuffle()


@_register_dataset("persona_based_chat", Topic.CREATIVE, "Persona-based chat")
def persona_based_chat() -> Dataset:
    """Persona-Based Chat 64K+."""
    data = load_dataset("nazlicanto/persona-based-chat")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"persona": [], "conv_id": []}

    for item in data:
        turns = item.get("dialogue", [])
        if not turns:
            continue

        dialogue = []
        if persona := item.get("persona_b"):
            persona_str = "\n".join(persona) if isinstance(persona, list) else str(persona)
            dialogue.append(Message("system", f"Persona: {persona_str}"))

        for i, turn in enumerate(turns):
            dialogue.append(Message("user" if i % 2 == 0 else "assistant", str(turn)))

        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)
            metadata["persona"].append(item.get("persona_b"))
            metadata["conv_id"].append(item.get("conv_id"))

    return Dataset(dialogues, labels, "persona_based_chat", metadata).shuffle()


@_register_dataset("roleplay", Topic.CREATIVE, "Roleplay")
def roleplay() -> Dataset:
    """Roleplay 5K+ character conversations."""
    data = load_dataset("hieunguyenminh/roleplay")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"character": []}

    for item in data:
        dialogue = []
        if system := item.get("system", item.get("instruction", "")):
            dialogue.append(Message("system", system))
        if convs := item.get("conversations", []):
            dialogue.extend(build_from_messages(convs))

        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)
            metadata["character"].append(item.get("character"))

    return Dataset(dialogues, labels, "roleplay", metadata).shuffle()


@_register_dataset("multi_character_dialogue", Topic.CREATIVE, "Multi-character dialogue")
def multi_character_dialogue() -> Dataset:
    """Multi-Character Dialogue 10K+ scenarios."""
    data = load_dataset("agentlans/multi-character-dialogue")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"genre": []}

    for item in data:
        text = item.get("dialogue", "")
        if not text:
            continue

        dialogues.append([Message("assistant", text)])
        labels.append(Label.NEGATIVE)
        metadata["genre"].append(item.get("genre") or item.get("category"))

    return Dataset(dialogues, labels, "multi_character_dialogue", metadata).shuffle()


@_register_dataset("literary_genre", Topic.CREATIVE, "Literary genre")
def literary_genre() -> Dataset:
    """Literary Genre Examples 86 genres."""
    data = load_dataset("agentlans/literary-genre-examples")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"genre": []}

    for item in data:
        genre = item.get("genre") or item.get("category") or ""
        example = item.get("example") or item.get("text") or item.get("content") or ""
        if not example:
            continue

        dialogues.append([Message("user", f"Write an example of {genre} writing."), Message("assistant", example)])
        labels.append(Label.NEGATIVE)
        metadata["genre"].append(genre)

    return Dataset(dialogues, labels, "literary_genre", metadata).shuffle()


@_register_dataset("writing_prompts", Topic.CREATIVE, "Writing prompts")
def writing_prompts() -> Dataset:
    """Writing Prompts for story generation."""
    data = load_dataset("fabraz/writingPromptAug")["train"]

    dialogues, labels = [], []

    for item in data:
        prompt = item.get("prompt") or item.get("input") or ""
        story = item.get("story") or item.get("output") or item.get("response") or ""
        if not prompt:
            continue

        dialogue = [Message("user", prompt)]
        if story:
            dialogue.append(Message("assistant", story))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "writing_prompts").shuffle()
