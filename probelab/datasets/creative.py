"""
Creative writing and roleplay datasets.

These datasets provide persona-based conversations, roleplay scenarios,
and creative writing examples, useful for training probes to detect
creative/roleplay content in model outputs.
"""

from ..types import Dialogue, Message
from .builders import build_from_messages
from .hf_dataset import DatasetSpec, HFDataset
from .registry import register


def _build_persona_chat(item: dict) -> Dialogue:
    """Builder for Synthetic Persona Chat format with persona context."""
    user1_personas = item.get("user 1 personas", "")
    user2_personas = item.get("user 2 personas", "")
    conversation = item.get("Best Generated Conversation", "")

    if not conversation:
        return []

    dialogue: Dialogue = []

    # Add persona context as system message
    if user1_personas or user2_personas:
        persona_context = f"User 1 personas: {user1_personas}\nUser 2 personas: {user2_personas}"
        dialogue.append(Message(role="system", content=persona_context))

    # Parse alternating turns "User 1: ... User 2: ..."
    parts = conversation.split("User ")
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("1:"):
            content = part[2:].strip()
            dialogue.append(Message(role="user", content=content))
        elif part.startswith("2:"):
            content = part[2:].strip()
            dialogue.append(Message(role="assistant", content=content))

    return dialogue


def _build_persona_based_chat(item: dict) -> Dialogue:
    """Builder for Persona-Based Chat with persona context and alternating turns."""
    persona = item.get("persona_b", [])
    dialogue_turns = item.get("dialogue", [])

    if not dialogue_turns:
        return []

    dialogue: Dialogue = []

    # Add persona as system message if available
    if persona:
        if isinstance(persona, list):
            persona_str = "\n".join(persona)
        else:
            persona_str = str(persona)
        dialogue.append(Message(role="system", content=f"Persona: {persona_str}"))

    # Parse dialogue turns - alternate between user and assistant
    for i, turn in enumerate(dialogue_turns):
        role = "user" if i % 2 == 0 else "assistant"
        content = turn if isinstance(turn, str) else str(turn)
        dialogue.append(Message(role=role, content=content))

    return dialogue


def _build_literary_genre(item: dict) -> Dialogue:
    """Builder for literary genre examples with synthetic user prompt."""
    genre = item.get("genre", item.get("category", ""))
    example = item.get("example", item.get("text", item.get("content", "")))

    if not example:
        return []

    return [
        Message(role="user", content=f"Write an example of {genre} writing."),
        Message(role="assistant", content=example),
    ]


def _build_roleplay(item: dict) -> Dialogue:
    """Builder for roleplay with system message and conversation."""
    conversations = item.get("conversations", [])
    system_msg = item.get("system", item.get("instruction", ""))

    dialogue: Dialogue = []

    if system_msg:
        dialogue.append(Message(role="system", content=system_msg))

    if conversations:
        dialogue.extend(build_from_messages(conversations))

    return dialogue


@register("creative", "Synthetic persona chat")
class SyntheticPersonaChatDataset(HFDataset):
    """
    Synthetic Persona Chat dataset from Google.

    Contains 20K+ persona-grounded conversations where each participant
    has defined personality traits that influence their dialogue.

    Source: https://huggingface.co/datasets/google/Synthetic-Persona-Chat

    Fields:
        - user 1 personas: Character traits for first participant
        - user 2 personas: Character traits for second participant
        - Best Generated Conversation: The dialogue exchange

    Metadata fields:
        - user1_personas: First participant's personas
        - user2_personas: Second participant's personas
    """

    base_name = "synthetic_persona_chat"
    spec = DatasetSpec(
        hf_path="google/Synthetic-Persona-Chat",
        shape="custom",
        builder_fn=_build_persona_chat,
        metadata_fields={
            "user1_personas": ("user 1 personas",),
            "user2_personas": ("user 2 personas",),
        },
    )


@register("creative", "Persona-based chat")
class PersonaBasedChatDataset(HFDataset):
    """
    Persona-Based Chat dataset.

    Contains 64K+ conversations where personality traits influence
    conversational flow between two personas.

    Source: https://huggingface.co/datasets/nazlicanto/persona-based-chat

    Fields:
        - persona_b: Character traits for participant
        - dialogue: Back-and-forth exchange

    Metadata fields:
        - persona: The persona traits
        - conv_id: Conversation identifier
    """

    base_name = "persona_based_chat"
    spec = DatasetSpec(
        hf_path="nazlicanto/persona-based-chat",
        shape="custom",
        builder_fn=_build_persona_based_chat,
        metadata_fields={
            "persona": ("persona_b",),
            "conv_id": ("conv_id",),
        },
    )


@register("creative", "Roleplay conversations")
class RoleplayDataset(HFDataset):
    """
    Roleplay dataset with character-based conversations.

    Contains 5K+ character roleplay interactions with defined
    personas and multi-turn exchanges.

    Source: https://huggingface.co/datasets/hieunguyenminh/roleplay

    Metadata fields:
        - character: Character name/description
    """

    base_name = "roleplay"
    spec = DatasetSpec(
        hf_path="hieunguyenminh/roleplay",
        shape="custom",
        builder_fn=_build_roleplay,
        metadata_fields={
            "character": ("character",),
        },
    )


@register("creative", "Multi-character dialogue")
class MultiCharacterDialogueDataset(HFDataset):
    """
    Multi-Character Dialogue dataset.

    Contains 10K+ multi-character dialogue scenarios spanning
    various genres (fantasy, sci-fi, noir, etc.).

    Source: https://huggingface.co/datasets/agentlans/multi-character-dialogue

    Metadata fields:
        - genre: Literary genre of the dialogue
    """

    base_name = "multi_character_dialogue"
    spec = DatasetSpec(
        hf_path="agentlans/multi-character-dialogue",
        shape="text",
        text_field="dialogue",
        text_as_assistant=True,
        metadata_fields={
            "genre": ("genre", "category"),
        },
    )


@register("creative", "Literary genre examples")
class LiteraryGenreDataset(HFDataset):
    """
    Literary Genre Examples dataset.

    Contains 86 fiction/nonfiction genre examples with
    representative paragraphs illustrating each genre's style.

    Source: https://huggingface.co/datasets/agentlans/literary-genre-examples

    Metadata fields:
        - genre: The literary genre classification
    """

    base_name = "literary_genre"
    spec = DatasetSpec(
        hf_path="agentlans/literary-genre-examples",
        shape="custom",
        builder_fn=_build_literary_genre,
        metadata_fields={
            "genre": ("genre", "category"),
        },
    )


@register("creative", "Writing prompts")
class WritingPromptsDataset(HFDataset):
    """
    Writing Prompts dataset for story generation.

    Contains writing prompts with generated story responses.

    Source: https://huggingface.co/datasets/fabraz/writingPromptAug
    """

    base_name = "writing_prompts"
    spec = DatasetSpec(
        hf_path="fabraz/writingPromptAug",
        shape="fields",
        user_fields=("prompt", "input"),
        assistant_fields=("story", "output", "response"),
    )
