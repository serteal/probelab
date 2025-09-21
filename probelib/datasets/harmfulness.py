from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ..types import DialogueDataType, Label, Message
from .base import DialogueDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def sanitize_prompt(prompt: str) -> str:
    return prompt.replace("<bos><start_of_turn>user\n", "").replace(
        "<end_of_turn>\n<start_of_turn>model\n", ""
    )


class WildJailbreakDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        data_path = REPO_ROOT / "data" / "harmfulness"
        df = pd.read_csv(data_path / "wildjailbreak_train.csv")
        filtered_harmful_df = df[df["data_type"] == "adversarial_harmful"]
        filtered_benign_df = df[df["data_type"] == "adversarial_benign"]

        dialogues = []
        labels = []

        for _, row in filtered_harmful_df.iterrows():
            dialogues.append(
                [
                    Message(role="user", content=str(row["adversarial"])),
                    Message(role="assistant", content=str(row["completion"])),
                ]
            )
            labels.append(Label.POSITIVE)

        for _, row in filtered_benign_df.iterrows():
            dialogues.append(
                [
                    Message(role="user", content=str(row["adversarial"])),
                    Message(role="assistant", content=str(row["completion"])),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


class CircuitBreakersDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
        if kwargs.get("split") == "test":
            dataset = dataset["circuit_breakers_test"]
        else:
            dataset = dataset["circuit_breakers_train"]

        dialogues = []
        labels = []

        for d in dataset:
            dialogues.append(
                [
                    Message(
                        role="user",
                        content=str(sanitize_prompt(d["prompt"])),
                    ),
                    Message(
                        role="assistant",
                        content=str(d["completion"]),
                    ),
                ]
            )
            labels.append(Label.POSITIVE)

        return dialogues, labels, None


class BenignInstructionsDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
        if kwargs.get("split") == "test":
            dataset = dataset["benign_instructions_test"]
        else:
            dataset = dataset["benign_instructions_train"]

        # Limit samples if requested
        max_samples = kwargs.get("max_samples")
        if max_samples and len(dataset) > max_samples:
            import random

            random.seed(42)
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)

        dialogues = []
        labels = []

        for d in dataset:
            dialogues.append(
                [
                    Message(
                        role="user",
                        content=str(sanitize_prompt(d["prompt"])),
                    ),
                    Message(
                        role="assistant",
                        content=str(d["completion"]),
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


class WildGuardMixDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # allenai/wildguardmix
        raise NotImplementedError("WildGuardMixDataset is not implemented yet.")


class XSTestResponseDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # allenai/xstest-response
        dataset = load_dataset("allenai/xstest-response")
        dialogues = []
        labels = []
        for d in dataset["response_harmfulness"]:
            dialogues.append(
                [
                    Message(role="user", content=str(d["prompt"])),
                    Message(
                        role="assistant",
                        content=str(d["response"]),
                    ),
                ]
            )
            # TODO: have analysed and this seems correct, but needs further check
            labels.append(Label.POSITIVE if d["label"] == "harmful" else Label.NEGATIVE)
        return dialogues, labels, None


class CoconotDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # allenai/coconot
        raise NotImplementedError("CoconotDataset is not implemented yet.")


class ToxicChatDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # lmsys/toxic-chat
        raise NotImplementedError("ToxicChatDataset is not implemented yet.")


class ClearHarmLlama3Dataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        data_path = REPO_ROOT / "data" / "harmfulness"
        df = pd.read_json(data_path / "outputs_huihui-ai_Meta-Llama-3.json")

        # Limit samples if requested
        max_samples = kwargs.get("max_samples")
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        dialogues = []
        labels = []

        for _, row in df.iterrows():
            dialogues.append(
                [
                    Message(role="user", content=str(row["prompt"])),
                    Message(
                        role="assistant",
                        content=str(row["generated_text"]),
                    ),
                ]
            )
            labels.append(Label.POSITIVE)

        return dialogues, labels, None


class ClearHarmMistralSmallDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        data_path = REPO_ROOT / "data" / "harmfulness"
        df = pd.read_json(
            data_path
            / "outputs_huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated_20250702_233538.json"
        )

        # Limit samples if requested
        max_samples = kwargs.get("max_samples")
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        dialogues = []
        labels = []

        for _, row in df.iterrows():
            dialogues.append(
                [
                    Message(role="user", content=str(row["prompt"])),
                    Message(
                        role="assistant",
                        content=str(row["generated_text"]),
                    ),
                ]
            )
            labels.append(Label.POSITIVE)

        return dialogues, labels, None
