from datasets import load_dataset

from ..types import DialogueDataType, Label, Message
from .base import DialogueDataset


class UltraChatDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("causal-lm/ultrachat")

        # Limit samples if requested
        max_samples = kwargs.get("max_samples")
        if max_samples and len(dataset["train"]) > max_samples:
            import random

            random.seed(42)
            indices = random.sample(range(len(dataset["train"])), max_samples)
            dataset["train"] = dataset["train"].select(indices)

        dialogues = []
        labels = []
        for element in dataset["train"]:
            dialogues.append(
                [
                    Message(
                        content=element["instruction"],
                        role="user",
                    ),
                    Message(
                        content=element["output"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


class LmsysChatDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # lmsys/lmsys-chat-1m
        raise NotImplementedError("LmsysChatDataset is not implemented yet.")


class AlpacaDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("yahma/alpaca-cleaned")

        dialogues = []
        labels = []
        for element in dataset["train"]:
            dialogues.append(
                [
                    Message(
                        content=element["instruction"],
                        role="user",
                    ),
                    Message(
                        content=element["output"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


class GPQAReasoningDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # spawn99/GPQA-diamond-ClaudeR1
        raise NotImplementedError("GPQAReasoningDataset is not implemented yet.")


class AlpacaFrenchDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # tbboukhari/Alpaca_french_instruct
        raise NotImplementedError("AlpacaFrenchDataset is not implemented yet.")


class GenericSpanishDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # FreedomIntelligence/evol-instruct-spanish, there's also in other langs
        dataset = load_dataset("FreedomIntelligence/evol-instruct-spanish")

        dialogues = []
        labels = []
        for element in dataset["train"]:
            dialogues.append(
                [
                    Message(
                        content=element["conversations"][0]["value"],
                        role="user",
                    ),
                    Message(
                        content=element["conversations"][1]["value"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


class MathInstructDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # TIGER-Lab/MathInstruct
        dataset = load_dataset("TIGER-Lab/MathInstruct")

        dialogues = []
        labels = []
        for element in dataset["train"]:
            dialogues.append(
                [
                    Message(
                        content=element["instruction"],
                        role="user",
                    ),
                    Message(
                        content=element["output"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


class MATHInstructionDataset(DialogueDataset):
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        # alpayariyak/MATH_Instruction_Format
        raise NotImplementedError("MATHInstructionDataset is not implemented yet.")
