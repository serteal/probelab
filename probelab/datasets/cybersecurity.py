"""Cybersecurity and defensive security datasets."""

import json
import logging
from typing import Any
from urllib.request import urlopen

import pandas as pd
from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset

logger = logging.getLogger(__name__)


@_register_dataset("trendyol_cybersecurity", Topic.CYBERSECURITY, "Trendyol cybersecurity")
def trendyol_cybersecurity() -> Dataset:
    """Trendyol Cybersecurity 53K covering 200+ domains."""
    data = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset")["train"]

    dialogues, labels = [], []

    for item in data:
        dialogue = []
        if system := item.get("system", ""):
            dialogue.append(Message("system", system))

        user = item.get("user", "")
        assistant = item.get("assistant", "")

        if user:
            dialogue.append(Message("user", user))
        if assistant:
            dialogue.append(Message("assistant", assistant))

        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "trendyol_cybersecurity").shuffle()


@_register_dataset("cybersecurity_dpo", Topic.CYBERSECURITY, "Cybersecurity DPO")
def cybersecurity_dpo(include_rejected: bool = True) -> Dataset:
    """Code Vulnerability Security DPO - secure vs insecure code pairs."""
    data = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"is_secure": []}

    for item in data:
        prompt = item.get("prompt", item.get("instruction", ""))
        if prompt and (chosen := item.get("chosen", "")):
            dialogues.append([Message("user", prompt), Message("assistant", chosen)])
            labels.append(Label.NEGATIVE)
            metadata["is_secure"].append(True)

        if include_rejected and prompt and (rejected := item.get("rejected", "")):
            dialogues.append([Message("user", prompt), Message("assistant", rejected)])
            labels.append(Label.POSITIVE)
            metadata["is_secure"].append(False)

    return Dataset(dialogues, labels, "cybersecurity_dpo", metadata).shuffle()


@_register_dataset("defensive_cybersecurity", Topic.CYBERSECURITY, "Defensive cybersecurity")
def defensive_cybersecurity() -> Dataset:
    """Defensive Cybersecurity V1 - 2.5K high-quality pairs."""
    data = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-v1")["train"]

    dialogues, labels = [], []

    for item in data:
        user = item.get("instruction") or item.get("input") or item.get("question") or ""
        assistant = item.get("response") or item.get("output") or item.get("answer") or ""
        if not user:
            continue

        dialogue = [Message("user", user)]
        if assistant:
            dialogue.append(Message("assistant", assistant))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "defensive_cybersecurity").shuffle()


@_register_dataset("wmdp_cyber", Topic.CYBERSECURITY, "WMDP cybersecurity MC questions")
def wmdp_cyber() -> Dataset:
    """WMDP-cyber: 1987 multiple-choice cybersecurity questions from CAIS."""
    data = load_dataset("cais/wmdp", "wmdp-cyber")["test"]

    dialogues, labels = [], []

    for item in data:
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        formatted = question + "\n\n" + "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        answer_text = f"The answer is {chr(65 + answer_idx)}: {choices[answer_idx]}"
        dialogues.append([Message("user", formatted), Message("assistant", answer_text)])
        labels.append(Label.POSITIVE)

    return Dataset(dialogues, labels, "wmdp_cyber").shuffle()


@_register_dataset("bigcodebench", Topic.CYBERSECURITY, "BigCodeBench coding tasks")
def bigcodebench(split: str = "v0.1.4") -> Dataset:
    """BigCodeBench: ~1140 coding tasks with tests.

    Args:
        split: Version split (default 'v0.1.4').
    """
    data = load_dataset("bigcode/bigcodebench", split=split)

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"task_id": [], "libs": []}

    for item in data:
        prompt = item.get("instruct_prompt") or item.get("complete_prompt", "")
        solution = item.get("canonical_solution", "")
        if not prompt:
            continue

        dialogue = [Message("user", prompt)]
        if solution:
            dialogue.append(Message("assistant", solution))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["task_id"].append(item.get("task_id"))
        metadata["libs"].append(item.get("libs"))

    return Dataset(dialogues, labels, "bigcodebench", metadata).shuffle()


@_register_dataset("cyberseceval", Topic.CYBERSECURITY, "CyberSecEval vulnerability detection")
def cyberseceval(config: str = "instruct") -> Dataset:
    """Meta CyberSecEval: code with CWE vulnerabilities.

    Args:
        config: 'autocomplete' or 'instruct' (default 'instruct').
    """
    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"cwe_identifier": [], "language": []}

    for lang in ["python", "php", "javascript", "rust", "java", "cpp", "c", "csharp"]:
        try:
            data = load_dataset("walledai/CyberSecEval", config, split=lang)
        except Exception:
            continue

        for item in data:
            prompt = item.get("prompt", "")
            if not prompt:
                continue

            dialogue = [Message("user", prompt)]
            dialogues.append(dialogue)
            labels.append(Label.POSITIVE)
            metadata["cwe_identifier"].append(item.get("cwe_identifier"))
            metadata["language"].append(lang)

    return Dataset(dialogues, labels, f"cyberseceval:{config}", metadata).shuffle()


@_register_dataset("cysecbench", Topic.CYBERSECURITY, "CySecBench cybersecurity prompts")
def cysecbench(path: str) -> Dataset:
    """CySecBench: 12K+ cybersecurity jailbreaking prompts.

    Args:
        path: Path to cysecbench.csv file.
    """
    df = pd.read_csv(path)

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"category": []}

    text_col = "Prompt" if "Prompt" in df.columns else df.columns[0]
    cat_col = "Category" if "Category" in df.columns else None

    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue
        dialogues.append([Message("user", text)])
        labels.append(Label.POSITIVE)
        metadata["category"].append(str(row[cat_col]) if cat_col else "")

    return Dataset(dialogues, labels, "cysecbench", metadata).shuffle()


@_register_dataset("intercode_ctf", Topic.CYBERSECURITY, "InterCode CTF challenges")
def intercode_ctf() -> Dataset:
    """InterCode CTF: 100 CTF challenges from picoCTF (GitHub)."""
    url = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/ctf/ic_ctf.json"
    data = json.loads(urlopen(url).read())

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"task_id": [], "tags": []}

    for item in data:
        query = item.get("query", "")
        if not query:
            continue

        dialogues.append([Message("user", query)])
        labels.append(Label.NEGATIVE)
        metadata["task_id"].append(item.get("task_id"))
        metadata["tags"].append(item.get("tags", []))

    return Dataset(dialogues, labels, "intercode_ctf", metadata).shuffle()


@_register_dataset("hackaprompt", Topic.CYBERSECURITY, "HackAPrompt adversarial prompts")
def hackaprompt() -> Dataset:
    """HackAPrompt: 600K+ adversarial prompt injection attempts."""
    data = load_dataset("hackaprompt/hackaprompt-dataset")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"level": [], "model": [], "correct": []}

    for item in data:
        user_input = item.get("user_input", "")
        if not user_input or not user_input.strip():
            continue

        dialogue = [Message("user", user_input)]
        completion = item.get("completion", "")
        if completion:
            dialogue.append(Message("assistant", completion))

        dialogues.append(dialogue)
        correct = item.get("correct", False)
        labels.append(Label.POSITIVE if correct else Label.NEGATIVE)
        metadata["level"].append(item.get("level"))
        metadata["model"].append(item.get("model"))
        metadata["correct"].append(correct)

    return Dataset(dialogues, labels, "hackaprompt", metadata).shuffle()


@_register_dataset("cyberrisk_qa", Topic.CYBERSECURITY, "CyberRisk Q/A pairs")
def cyberrisk_qa() -> Dataset:
    """CyberRisk 14K+ offensive cybersecurity Q/A pairs (augmented, multilingual)."""
    try:
        hf_ds = load_dataset("serteal/cyberrisk-qa-pairs", split="train")
    except Exception as e:
        logger.debug("HF load failed for serteal/cyberrisk-qa-pairs: %s", e)
        raise FileNotFoundError(
            "cyberrisk-qa-pairs not found on HuggingFace (serteal/cyberrisk-qa-pairs)"
        ) from e

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {
        "topic": [], "subtopic": [], "question_type": [],
        "augmentation": [], "language": [], "base_id": [],
    }

    for item in hf_ds:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question or not answer:
            continue
        dialogues.append([Message("user", question), Message("assistant", answer)])
        labels.append(Label.POSITIVE)
        metadata["topic"].append(item.get("topic", ""))
        metadata["subtopic"].append(item.get("subtopic", ""))
        metadata["question_type"].append(item.get("question_type", ""))
        metadata["augmentation"].append(item.get("augmentation", ""))
        metadata["language"].append(item.get("language", ""))
        metadata["base_id"].append(item.get("base_id", -1))

    return Dataset(dialogues, labels, "cyberrisk_qa", metadata).shuffle()
