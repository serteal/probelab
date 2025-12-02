"""Pre-built datasets for probe training and evaluation."""

from .base import DialogueDataset

# Deception detection datasets
from .deception import (
    AIAuditDataset,
    AILiarDataset,
    DolusChatDataset,
    InsiderTradingDataset,
    REPEDataset,
    RoleplayingDataset,
    SandbaggingDataset,
    TruthfulQADataset,
    WerewolfDataset,
)

# Harmfulness detection datasets
from .harmfulness import (
    BenignInstructionsDataset,
    CircuitBreakersDataset,
    ClearHarmLlama3Dataset,
    ClearHarmMistralSmallDataset,
    WildGuardMixDataset,
    WildJailbreakDataset,
    XSTestResponseDataset,
)

# General conversation datasets
from .ood import (
    AlpacaDataset,
    ChatbotArenaDataset,
    GenericSpanishDataset,
    LmsysChatDataset,
    MathInstructDataset,
    ShareGPTDataset,
    UltraChat200kDataset,
    UltraChatDataset,
)

# Multilingual datasets
from .multilingual import (
    MultilingualThinkingDataset,
    PaloMultilingualDataset,
    WildChatDataset,
)

# Medical datasets
from .medical import (
    ClinicalNotesDataset,
    KnowMedicalDialogueDataset,
    MedDialogDataset,
    MedicalDialogueSOAPDataset,
)

# Mental health datasets
from .mental_health import (
    EmotionalSupportDataset,
    MentalChatDataset,
    MentalHealthCounselingDataset,
    ProsocialDialogDataset,
)

# Cybersecurity datasets
from .cybersecurity import (
    CybersecurityDPODataset,
    DefensiveCybersecurityDataset,
    TrendyolCybersecurityDataset,
)

# Agent and function calling datasets
from .agents import (
    FunctionCallingShareGPTDataset,
    GlaiveFunctionCallingDataset,
    HermesFunctionCallingDataset,
    XLAMFunctionCallingDataset,
)

# Math and reasoning datasets
from .reasoning import (
    MathInstructEnhancedDataset,
    NuminaMathCoTDataset,
    OpenMathReasoningDataset,
    OpenR1MathDataset,
)

# Creative writing and roleplay datasets
from .creative import (
    LiteraryGenreDataset,
    MultiCharacterDialogueDataset,
    PersonaBasedChatDataset,
    RoleplayDataset,
    SyntheticPersonaChatDataset,
    WritingPromptsDataset,
)

# Legal and finance datasets
from .legal_finance import (
    CaselawDataset,
    FinancialPhrasebankDataset,
    FinanceTasksDataset,
    LegalAdviceRedditDataset,
)

# Science and STEM datasets
from .science import (
    BiochemReasoningDataset,
    BiologyToTDataset,
    MMMUDataset,
    ScienceQADataset,
    StemQADataset,
)

__all__ = [
    # Base
    "DialogueDataset",
    # Deception
    "AIAuditDataset",
    "AILiarDataset",
    "DolusChatDataset",
    "InsiderTradingDataset",
    "REPEDataset",
    "RoleplayingDataset",
    "SandbaggingDataset",
    "TruthfulQADataset",
    "WerewolfDataset",
    # Harmfulness
    "BenignInstructionsDataset",
    "CircuitBreakersDataset",
    "ClearHarmLlama3Dataset",
    "ClearHarmMistralSmallDataset",
    "WildGuardMixDataset",
    "WildJailbreakDataset",
    "XSTestResponseDataset",
    # General / OOD
    "AlpacaDataset",
    "ChatbotArenaDataset",
    "GenericSpanishDataset",
    "LmsysChatDataset",
    "MathInstructDataset",
    "ShareGPTDataset",
    "UltraChat200kDataset",
    "UltraChatDataset",
    # Multilingual
    "MultilingualThinkingDataset",
    "PaloMultilingualDataset",
    "WildChatDataset",
    # Medical
    "ClinicalNotesDataset",
    "KnowMedicalDialogueDataset",
    "MedDialogDataset",
    "MedicalDialogueSOAPDataset",
    # Mental Health
    "EmotionalSupportDataset",
    "MentalChatDataset",
    "MentalHealthCounselingDataset",
    "ProsocialDialogDataset",
    # Cybersecurity
    "CybersecurityDPODataset",
    "DefensiveCybersecurityDataset",
    "TrendyolCybersecurityDataset",
    # Agents / Function Calling
    "FunctionCallingShareGPTDataset",
    "GlaiveFunctionCallingDataset",
    "HermesFunctionCallingDataset",
    "XLAMFunctionCallingDataset",
    # Math / Reasoning
    "MathInstructEnhancedDataset",
    "NuminaMathCoTDataset",
    "OpenMathReasoningDataset",
    "OpenR1MathDataset",
    # Creative / Roleplay
    "LiteraryGenreDataset",
    "MultiCharacterDialogueDataset",
    "PersonaBasedChatDataset",
    "RoleplayDataset",
    "SyntheticPersonaChatDataset",
    "WritingPromptsDataset",
    # Legal / Finance
    "CaselawDataset",
    "FinancialPhrasebankDataset",
    "FinanceTasksDataset",
    "LegalAdviceRedditDataset",
    # Science / STEM
    "BiochemReasoningDataset",
    "BiologyToTDataset",
    "MMMUDataset",
    "ScienceQADataset",
    "StemQADataset",
]
