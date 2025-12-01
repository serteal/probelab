"""
Agent and function calling datasets.

These datasets provide tool use, function calling, and agentic conversations,
useful for training probes to detect agentic behavior in model outputs.
"""

import json

from ..types import Dialogue, Message
from .hf_dataset import DatasetSpec, HFDataset


def _build_xlam_function_call(item: dict) -> Dialogue:
    """Builder for xLAM function calling format with JSON tools/answers."""
    query = item.get("query", "")
    tools = item.get("tools", "")
    answers = item.get("answers", "")

    if not query:
        return []

    # Parse tools to create system message
    try:
        tools_obj = json.loads(tools) if isinstance(tools, str) else tools
        tools_str = json.dumps(tools_obj, indent=2)
    except (json.JSONDecodeError, TypeError):
        tools_str = str(tools)

    # Parse answers to create assistant response
    try:
        answers_obj = json.loads(answers) if isinstance(answers, str) else answers
        answers_str = json.dumps(answers_obj, indent=2)
    except (json.JSONDecodeError, TypeError):
        answers_str = str(answers)

    return [
        Message(
            role="system",
            content=f"You have access to the following tools:\n{tools_str}",
        ),
        Message(role="user", content=query),
        Message(role="assistant", content=f"Function calls:\n{answers_str}"),
    ]


def _build_glaive_conversation(item: dict) -> Dialogue:
    """Builder for Glaive's SYSTEM:/USER:/ASSISTANT: format."""
    text = item.get("sample", "")
    if not text:
        return []

    dialogue: Dialogue = []
    lines = text.split("\n")

    current_role = None
    current_content: list[str] = []

    role_map = {
        "SYSTEM:": "system",
        "USER:": "user",
        "ASSISTANT:": "assistant",
        "FUNCTION RESPONSE:": "user",  # Treat function response as user input
    }

    for line in lines:
        # Check if this line starts a new role
        found_role = None
        line_content = ""
        for prefix, role in role_map.items():
            if line.strip().startswith(prefix):
                found_role = role
                line_content = line.strip()[len(prefix) :].strip()
                break

        if found_role:
            # Save previous message if exists
            if current_role and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    dialogue.append(Message(role=current_role, content=content))

            current_role = found_role
            current_content = [line_content] if line_content else []
        else:
            # Continue current message
            if current_role:
                current_content.append(line)

    # Don't forget the last message
    if current_role and current_content:
        content = "\n".join(current_content).strip()
        if content:
            dialogue.append(Message(role=current_role, content=content))

    return dialogue


class XLAMFunctionCallingDataset(HFDataset):
    """
    xLAM Function Calling 60K dataset from Salesforce.

    Contains 60K function calling examples with queries, available tools,
    and structured answers showing which functions to call.

    Source: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

    Fields:
        - query: User's request
        - tools: Available functions with parameters
        - answers: Function calls with arguments

    Metadata fields:
        - tools: Raw tools JSON
        - answers: Raw answers JSON
    """

    base_name = "xlam_function_calling"
    spec = DatasetSpec(
        hf_path="Salesforce/xlam-function-calling-60k",
        shape="custom",
        builder_fn=_build_xlam_function_call,
        metadata_fields={
            "tools": ("tools",),
            "answers": ("answers",),
        },
    )


class GlaiveFunctionCallingDataset(HFDataset):
    """
    Glaive Function Calling dataset.

    Contains 52K conversational function calling examples with system prompts
    defining available functions.

    Source: https://huggingface.co/datasets/glaiveai/glaive-function-calling
    """

    base_name = "glaive_function_calling"
    spec = DatasetSpec(
        hf_path="glaiveai/glaive-function-calling",
        shape="custom",
        builder_fn=_build_glaive_conversation,
    )


class HermesFunctionCallingDataset(HFDataset):
    """
    Hermes Function Calling V1 dataset from NousResearch.

    Structured output dataset with function-calling conversations,
    JSON-mode, and agentic examples.

    Source: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
    """

    base_name = "hermes_function_calling"
    spec = DatasetSpec(
        hf_path="NousResearch/hermes-function-calling-v1",
        shape="messages",
        messages_field="conversations",
        role_field="role",
        content_field="content",
        role_fallback="from",
        content_fallback="value",
    )


class FunctionCallingShareGPTDataset(HFDataset):
    """
    Function Calling ShareGPT dataset.

    Contains 86K chat examples that include function calling with
    0-2 function definitions per conversation.

    Source: https://huggingface.co/datasets/hypervariance/function-calling-sharegpt
    """

    base_name = "function_calling_sharegpt"
    spec = DatasetSpec(
        hf_path="hypervariance/function-calling-sharegpt",
        shape="messages",
        messages_field="conversations",
        role_field="from",
        content_field="value",
        role_fallback="role",
        content_fallback="content",
    )
