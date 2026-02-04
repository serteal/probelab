"""Dataset registry - simple dict mapping names to loader functions."""

from .base import Dataset, LoaderFn

# Registry: name -> (loader_fn, category, description)
REGISTRY: dict[str, tuple[LoaderFn, str, str]] = {}


def _register(name: str, category: str, description: str = "") -> callable:
    """Register a loader function."""
    def decorator(fn: LoaderFn) -> LoaderFn:
        REGISTRY[name] = (fn, category, description)
        return fn
    return decorator


def load(name: str, **kwargs) -> Dataset:
    """Load a dataset by name."""
    if name not in REGISTRY:
        raise KeyError(f"Dataset '{name}' not found. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[name][0](**kwargs)


def list_datasets(category: str | None = None) -> list[str]:
    """List available dataset names, optionally filtered by category."""
    if category is None:
        return sorted(REGISTRY.keys())
    return sorted(n for n, (_, c, _) in REGISTRY.items() if c == category)


def list_categories() -> list[str]:
    """List available dataset categories."""
    return sorted({c for _, (_, c, _) in REGISTRY.items()})


def info(name: str) -> dict:
    """Get information about a dataset."""
    if name not in REGISTRY:
        raise KeyError(f"Dataset '{name}' not found. Available: {sorted(REGISTRY.keys())}")
    fn, category, description = REGISTRY[name]
    return {"name": name, "category": category, "description": description, "loader": fn}


def _init_registry() -> None:
    """Initialize registry by importing all dataset modules."""
    # Import modules to register their loaders
    from . import agents, creative, cybersecurity, deception, harmfulness, legal_finance, medical, mental_health, multilingual, ood, reasoning, science

    # Deception
    REGISTRY.update({
        "ai_audit": (deception.ai_audit, "deception", "AI audit dataset"),
        "ai_liar": (deception.ai_liar, "deception", "AI liar dataset"),
        "dolus_chat": (deception.dolus_chat, "deception", "Dolus chat deception"),
        "insider_trading": (deception.insider_trading, "deception", "Insider trading dataset"),
        "repe": (deception.repe, "deception", "REPE dataset"),
        "roleplaying": (deception.roleplaying, "deception", "Roleplaying dataset"),
        "sandbagging": (deception.sandbagging, "deception", "Sandbagging dataset"),
        "truthful_qa": (deception.truthful_qa, "deception", "TruthfulQA dataset"),
        "werewolf": (deception.werewolf, "deception", "Werewolf dataset"),
    })

    # Harmfulness
    REGISTRY.update({
        "circuit_breakers": (harmfulness.circuit_breakers, "harmfulness", "Circuit breakers dataset"),
        "benign_instructions": (harmfulness.benign_instructions, "harmfulness", "Benign instructions"),
        "wild_jailbreak": (harmfulness.wild_jailbreak, "harmfulness", "Wild jailbreak dataset"),
        "wildguard_mix": (harmfulness.wildguard_mix, "harmfulness", "WildGuard mix dataset"),
        "xstest_response": (harmfulness.xstest_response, "harmfulness", "XSTest response dataset"),
        "clearharm_llama3": (harmfulness.clearharm_llama3, "harmfulness", "ClearHarm Llama3"),
        "clearharm_mistral": (harmfulness.clearharm_mistral, "harmfulness", "ClearHarm Mistral"),
    })

    # OOD / General
    REGISTRY.update({
        "ultrachat": (ood.ultrachat, "ood", "UltraChat dataset"),
        "ultrachat_200k": (ood.ultrachat_200k, "ood", "UltraChat 200K"),
        "lmsys_chat": (ood.lmsys_chat, "ood", "LMSYS chat"),
        "sharegpt": (ood.sharegpt, "ood", "ShareGPT dataset"),
        "chatbot_arena": (ood.chatbot_arena, "ood", "Chatbot arena"),
        "alpaca": (ood.alpaca, "ood", "Alpaca dataset"),
        "generic_spanish": (ood.generic_spanish, "ood", "Generic Spanish"),
        "math_instruct": (ood.math_instruct, "ood", "Math instruct"),
    })

    # Multilingual
    REGISTRY.update({
        "wildchat": (multilingual.wildchat, "multilingual", "WildChat 1M"),
        "multilingual_thinking": (multilingual.multilingual_thinking, "multilingual", "Multilingual thinking"),
        "palo_multilingual": (multilingual.palo_multilingual, "multilingual", "PALO multilingual"),
    })

    # Medical
    REGISTRY.update({
        "meddialog": (medical.meddialog, "medical", "MedDialog"),
        "medical_soap": (medical.medical_soap, "medical", "Medical SOAP"),
        "clinical_notes": (medical.clinical_notes, "medical", "Clinical notes"),
        "know_medical": (medical.know_medical, "medical", "Know medical dialogue"),
    })

    # Mental health
    REGISTRY.update({
        "mentalchat": (mental_health.mentalchat, "mental_health", "MentalChat"),
        "mental_health_counseling": (mental_health.mental_health_counseling, "mental_health", "Mental health counseling"),
        "prosocial_dialog": (mental_health.prosocial_dialog, "mental_health", "Prosocial dialog"),
        "emotional_support": (mental_health.emotional_support, "mental_health", "Emotional support"),
    })

    # Cybersecurity
    REGISTRY.update({
        "trendyol_cybersecurity": (cybersecurity.trendyol_cybersecurity, "cybersecurity", "Trendyol cybersecurity"),
        "cybersecurity_dpo": (cybersecurity.cybersecurity_dpo, "cybersecurity", "Cybersecurity DPO"),
        "defensive_cybersecurity": (cybersecurity.defensive_cybersecurity, "cybersecurity", "Defensive cybersecurity"),
    })

    # Agents
    REGISTRY.update({
        "xlam_function_calling": (agents.xlam_function_calling, "agents", "xLAM function calling"),
        "glaive_function_calling": (agents.glaive_function_calling, "agents", "Glaive function calling"),
        "hermes_function_calling": (agents.hermes_function_calling, "agents", "Hermes function calling"),
        "function_calling_sharegpt": (agents.function_calling_sharegpt, "agents", "Function calling ShareGPT"),
    })

    # Reasoning
    REGISTRY.update({
        "openr1_math": (reasoning.openr1_math, "reasoning", "OpenR1 math"),
        "openmath_reasoning": (reasoning.openmath_reasoning, "reasoning", "OpenMath reasoning"),
        "numina_math": (reasoning.numina_math, "reasoning", "NuminaMath CoT"),
        "math_instruct_enhanced": (reasoning.math_instruct_enhanced, "reasoning", "MathInstruct enhanced"),
    })

    # Creative
    REGISTRY.update({
        "synthetic_persona_chat": (creative.synthetic_persona_chat, "creative", "Synthetic persona chat"),
        "persona_based_chat": (creative.persona_based_chat, "creative", "Persona-based chat"),
        "roleplay": (creative.roleplay, "creative", "Roleplay"),
        "multi_character_dialogue": (creative.multi_character_dialogue, "creative", "Multi-character dialogue"),
        "literary_genre": (creative.literary_genre, "creative", "Literary genre"),
        "writing_prompts": (creative.writing_prompts, "creative", "Writing prompts"),
    })

    # Legal/Finance
    REGISTRY.update({
        "caselaw": (legal_finance.caselaw, "legal_finance", "Case law"),
        "finance_tasks": (legal_finance.finance_tasks, "legal_finance", "Finance tasks"),
        "financial_phrasebank": (legal_finance.financial_phrasebank, "legal_finance", "Financial phrasebank"),
        "legal_advice_reddit": (legal_finance.legal_advice_reddit, "legal_finance", "Legal advice Reddit"),
    })

    # Science
    REGISTRY.update({
        "scienceqa": (science.scienceqa, "science", "ScienceQA"),
        "mmmu": (science.mmmu, "science", "MMMU"),
        "biology_tot": (science.biology_tot, "science", "Biology ToT"),
        "biochem_reasoning": (science.biochem_reasoning, "science", "Biochem reasoning"),
        "stem_qa": (science.stem_qa, "science", "STEM Q&A"),
    })


_init_registry()
