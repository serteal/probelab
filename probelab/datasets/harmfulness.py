"""Harmfulness detection datasets."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _sanitize_prompt(prompt: str) -> str:
    return prompt.replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model\n", "")


@_register_dataset("wild_jailbreak", Topic.HARMFULNESS, "Wild jailbreak dataset")
def wild_jailbreak() -> Dataset:
    df = pd.read_csv(REPO_ROOT / "data" / "harmfulness" / "wildjailbreak_train.csv")
    dialogues, labels = [], []

    for _, row in df[df["data_type"] == "adversarial_harmful"].iterrows():
        dialogues.append([Message("user", str(row["adversarial"])), Message("assistant", str(row["completion"]))])
        labels.append(Label.POSITIVE)

    for _, row in df[df["data_type"] == "adversarial_benign"].iterrows():
        dialogues.append([Message("user", str(row["adversarial"])), Message("assistant", str(row["completion"]))])
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "wild_jailbreak").shuffle()


@_register_dataset("circuit_breakers", Topic.HARMFULNESS, "Circuit breakers dataset")
def circuit_breakers(split: str = "train") -> Dataset:
    ds = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
    data = ds["circuit_breakers_test"] if split == "test" else ds["circuit_breakers_train"]

    dialogues = [[Message("user", _sanitize_prompt(d["prompt"])), Message("assistant", d["completion"])] for d in data]
    labels = [Label.POSITIVE] * len(dialogues)

    return Dataset(dialogues, labels, "circuit_breakers").shuffle()


@_register_dataset("benign_instructions", Topic.HARMFULNESS, "Benign instructions")
def benign_instructions(split: str = "train") -> Dataset:
    ds = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
    data = ds["benign_instructions_test"] if split == "test" else ds["benign_instructions_train"]

    dialogues = [[Message("user", _sanitize_prompt(d["prompt"])), Message("assistant", d["completion"])] for d in data]
    labels = [Label.NEGATIVE] * len(dialogues)

    return Dataset(dialogues, labels, "benign_instructions").shuffle()


@_register_dataset("wildguard_mix", Topic.HARMFULNESS, "WildGuard mix dataset")
def wildguard_mix(split: str = "train") -> Dataset:
    ds_split = "wildguardtest" if split == "test" else "wildguardtrain"
    data = load_dataset("allenai/wildguardmix", ds_split)[split]

    dialogues, labels = [], []
    metadata = {"prompt_harm_label": [], "response_refusal_label": [], "response_harm_label": [], "adversarial": [], "subcategory": [], "response": []}

    for d in data:
        dialogues.append([Message("system", "You are a helpful, honest assistant."), Message("user", str(d["prompt"]))])
        labels.append(Label.POSITIVE if str(d["prompt_harm_label"]).lower() == "harmful" else Label.NEGATIVE)
        for k in metadata:
            metadata[k].append(d.get(k))

    return Dataset(dialogues, labels, "wildguard_mix", metadata).shuffle()


@_register_dataset("xstest_response", Topic.HARMFULNESS, "XSTest response dataset")
def xstest_response() -> Dataset:
    data = load_dataset("allenai/xstest-response")["response_harmfulness"]

    dialogues = [[Message("user", str(d["prompt"])), Message("assistant", str(d["response"]))] for d in data]
    labels = [Label.POSITIVE if d["label"] == "harmful" else Label.NEGATIVE for d in data]

    return Dataset(dialogues, labels, "xstest_response").shuffle()


@_register_dataset("clearharm_llama3", Topic.HARMFULNESS, "ClearHarm Llama3")
def clearharm_llama3(path: str | None = None) -> Dataset:
    json_path = Path(path) if path else REPO_ROOT / "data" / "harmfulness" / "outputs_huihui-ai_Meta-Llama-3.json"
    df = pd.read_json(json_path)

    dialogues = [[Message("user", str(row["prompt"])), Message("assistant", str(row["generated_text"]))] for _, row in df.iterrows()]
    labels = [Label.POSITIVE] * len(dialogues)

    return Dataset(dialogues, labels, "clearharm_llama3").shuffle()


@_register_dataset("clearharm_mistral", Topic.HARMFULNESS, "ClearHarm Mistral")
def clearharm_mistral(path: str | None = None) -> Dataset:
    json_path = Path(path) if path else REPO_ROOT / "data" / "harmfulness" / "outputs_huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated_20250702_233538.json"
    df = pd.read_json(json_path)

    dialogues = [[Message("user", str(row["prompt"])), Message("assistant", str(row["generated_text"]))] for _, row in df.iterrows()]
    labels = [Label.POSITIVE] * len(dialogues)

    return Dataset(dialogues, labels, "clearharm_mistral").shuffle()


@_register_dataset("clearharm_qwen3", Topic.HARMFULNESS, "ClearHarm Qwen3")
def clearharm_qwen3(path: str | None = None) -> Dataset:
    json_path = Path(path) if path else REPO_ROOT / "data" / "harmfulness" / "outputs_huihui-ai_Huihui-Qwen3-14B-abliterated-v2_20250702_212758_cleaned.json"
    df = pd.read_json(json_path)

    dialogues = [[Message("user", str(row["prompt"])), Message("assistant", str(row["generated_text"]))] for _, row in df.iterrows()]
    labels = [Label.POSITIVE] * len(dialogues)

    return Dataset(dialogues, labels, "clearharm_qwen3").shuffle()


@_register_dataset("or_bench", Topic.HARMFULNESS, "OR-Bench over-refusal benchmark")
def or_bench(config: str = "or-bench-80k") -> Dataset:
    """Load OR-Bench dataset (over-refusal benchmark).

    Args:
        config: HF config name. One of:
            - 'or-bench-80k': 80K safe prompts (Label.NEGATIVE)
            - 'or-bench-hard-1k': 1K hard safe prompts (Label.NEGATIVE)
            - 'or-bench-toxic': 655 toxic prompts (Label.POSITIVE)
    """
    data = load_dataset("bench-llms/or-bench", config)["train"]

    label = Label.POSITIVE if config == "or-bench-toxic" else Label.NEGATIVE
    dialogues = [[Message("user", str(row["prompt"]))] for row in data]
    labels = [label] * len(dialogues)
    metadata = {"category": [row.get("category", "") for row in data]}

    return Dataset(dialogues, labels, f"or_bench:{config}", metadata).shuffle()
