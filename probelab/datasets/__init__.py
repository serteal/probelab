"""Pre-built datasets for probe training and evaluation.

Registry API (recommended):
    load(name)          - Load dataset by name
    list_datasets()     - List available dataset names
    list_categories()   - List available categories
    info(name)          - Get dataset information

Example:
    >>> import probelab as pl
    >>> dataset = pl.datasets.load("circuit_breakers")
    >>> pl.datasets.list_datasets(category="deception")
    ['ai_audit', 'ai_liar', 'dolus_chat', ...]
"""

from .base import Dataset
from .registry import info, list_categories, list_datasets, load

# Deception
from .deception import ai_audit, ai_liar, dolus_chat, insider_trading, repe, roleplaying, sandbagging, truthful_qa, werewolf

# Harmfulness
from .harmfulness import benign_instructions, circuit_breakers, clearharm_llama3, clearharm_mistral, wild_jailbreak, wildguard_mix, xstest_response

# OOD / General
from .ood import alpaca, chatbot_arena, generic_spanish, lmsys_chat, math_instruct, sharegpt, ultrachat, ultrachat_200k

# Multilingual
from .multilingual import multilingual_thinking, palo_multilingual, wildchat

# Medical
from .medical import clinical_notes, know_medical, meddialog, medical_soap

# Mental health
from .mental_health import emotional_support, mentalchat, mental_health_counseling, prosocial_dialog

# Cybersecurity
from .cybersecurity import cybersecurity_dpo, defensive_cybersecurity, trendyol_cybersecurity

# Agents
from .agents import function_calling_sharegpt, glaive_function_calling, hermes_function_calling, xlam_function_calling

# Reasoning
from .reasoning import math_instruct_enhanced, numina_math, openmath_reasoning, openr1_math

# Creative
from .creative import literary_genre, multi_character_dialogue, persona_based_chat, roleplay, synthetic_persona_chat, writing_prompts

# Legal/Finance
from .legal_finance import caselaw, finance_tasks, financial_phrasebank, legal_advice_reddit

# Science
from .science import biochem_reasoning, biology_tot, mmmu, scienceqa, stem_qa

__all__ = [
    # Registry API
    "load",
    "list_datasets",
    "list_categories",
    "info",
    # Base
    "Dataset",
    # Deception
    "ai_audit",
    "ai_liar",
    "dolus_chat",
    "insider_trading",
    "repe",
    "roleplaying",
    "sandbagging",
    "truthful_qa",
    "werewolf",
    # Harmfulness
    "benign_instructions",
    "circuit_breakers",
    "clearharm_llama3",
    "clearharm_mistral",
    "wild_jailbreak",
    "wildguard_mix",
    "xstest_response",
    # OOD / General
    "alpaca",
    "chatbot_arena",
    "generic_spanish",
    "lmsys_chat",
    "math_instruct",
    "sharegpt",
    "ultrachat",
    "ultrachat_200k",
    # Multilingual
    "multilingual_thinking",
    "palo_multilingual",
    "wildchat",
    # Medical
    "clinical_notes",
    "know_medical",
    "meddialog",
    "medical_soap",
    # Mental health
    "emotional_support",
    "mentalchat",
    "mental_health_counseling",
    "prosocial_dialog",
    # Cybersecurity
    "cybersecurity_dpo",
    "defensive_cybersecurity",
    "trendyol_cybersecurity",
    # Agents
    "function_calling_sharegpt",
    "glaive_function_calling",
    "hermes_function_calling",
    "xlam_function_calling",
    # Reasoning
    "math_instruct_enhanced",
    "numina_math",
    "openmath_reasoning",
    "openr1_math",
    # Creative
    "literary_genre",
    "multi_character_dialogue",
    "persona_based_chat",
    "roleplay",
    "synthetic_persona_chat",
    "writing_prompts",
    # Legal/Finance
    "caselaw",
    "finance_tasks",
    "financial_phrasebank",
    "legal_advice_reddit",
    # Science
    "biochem_reasoning",
    "biology_tot",
    "mmmu",
    "scienceqa",
    "stem_qa",
]
