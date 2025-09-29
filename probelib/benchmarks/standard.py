"""Standard benchmark definitions and registry."""

import probelib as pl
from probelib.benchmarks.core import Benchmark
from probelib.benchmarks.specs import EvalSpec, TrainSpec


def deception_suite(model: str = "meta-llama/Llama-2-7b-chat-hf") -> Benchmark:
    """
    Standard deception detection benchmark suite.

    Trains on AI Audit, AI Liar, and REPE datasets, then evaluates on their
    test sets plus TruthfulQA as an out-of-distribution benchmark.

    Args:
        model: Model name/path to use (default: Llama-2-7b-chat)

    Returns:
        Configured Benchmark ready to run
    """
    return Benchmark(
        name="Deception Detection Suite v1",
        description="Multi-dataset deception detection with OOD evaluation",
        default_model=model,
        training=TrainSpec(
            datasets=[
                pl.datasets.AIAuditDataset(split="train"),
                pl.datasets.AILiarDataset(split="train"),
                pl.datasets.REPEDataset(split="train"),
            ],
            mask=pl.masks.assistant(),
            probe_config={"layer": 16, "sequence_aggregation": "mean"},
        ),
        evaluations=[
            EvalSpec(
                dataset=pl.datasets.AIAuditDataset(split="test"),
                mask=pl.masks.assistant(),
                name="AI Audit",
            ),
            EvalSpec(
                dataset=pl.datasets.AILiarDataset(split="test"),
                mask=pl.masks.assistant(),
                name="AI Liar",
            ),
            EvalSpec(
                dataset=pl.datasets.REPEDataset(split="test"),
                mask=pl.masks.assistant(),
                name="REPE",
            ),
            EvalSpec(
                dataset=pl.datasets.TruthfulQADataset(split="test"),
                mask=pl.masks.assistant(),
                name="TruthfulQA (OOD)",
            ),
        ],
    )


def harmfulness_suite(model: str = "meta-llama/Llama-2-7b-chat-hf") -> Benchmark:
    """
    Standard harmfulness detection benchmark suite.

    Trains on Circuit Breakers and WildGuardMix, evaluates on multiple
    harmfulness datasets including XSTest and ToxicChat.

    Args:
        model: Model name/path to use (default: Llama-2-7b-chat)

    Returns:
        Configured Benchmark ready to run
    """
    return Benchmark(
        name="Harmfulness Detection Suite v1",
        description="Multi-dataset harmfulness detection benchmark",
        default_model=model,
        training=TrainSpec(
            datasets=[
                pl.datasets.CircuitBreakersDataset(split="train"),
                pl.datasets.WildGuardMixDataset(split="train"),
            ],
            mask=pl.masks.assistant(),
            probe_config={"layer": 16, "sequence_aggregation": "mean"},
        ),
        evaluations=[
            EvalSpec(
                dataset=pl.datasets.CircuitBreakersDataset(split="test"),
                mask=pl.masks.assistant(),
                name="Circuit Breakers",
            ),
            EvalSpec(
                dataset=pl.datasets.WildGuardMixDataset(split="test"),
                mask=pl.masks.assistant(),
                name="WildGuard Mix",
            ),
            EvalSpec(
                dataset=pl.datasets.XSTestResponseDataset(split="test"),
                mask=pl.masks.assistant(),
                name="XSTest",
            ),
            EvalSpec(
                dataset=pl.datasets.ToxicChatDataset(split="test"),
                mask=pl.masks.assistant(),
                name="ToxicChat",
            ),
        ],
    )


def cross_model_deception(
    train_model: str = "meta-llama/Llama-2-7b-chat-hf",
) -> Benchmark:
    """
    Cross-model generalization benchmark for deception detection.

    Trains on Llama-2, evaluates on Llama-2, Gemma, and Mistral to test
    whether learned representations transfer across model families.

    Args:
        train_model: Model to train on (default: Llama-2-7b-chat)

    Returns:
        Configured Benchmark ready to run

    Warning:
        Cross-model evaluation requires activations from the same layer
        across different architectures. Results may not be directly comparable.
    """
    return Benchmark(
        name="Cross-Model Deception Generalization",
        description="Test probe generalization across model families",
        training=TrainSpec(
            datasets=[pl.datasets.AIAuditDataset(split="train")],
            model=train_model,
            mask=pl.masks.assistant(),
            probe_config={"layer": 16, "sequence_aggregation": "mean"},
        ),
        evaluations=[
            EvalSpec(
                dataset=pl.datasets.AIAuditDataset(split="test"),
                model="meta-llama/Llama-2-7b-chat-hf",
                mask=pl.masks.assistant(),
                name="Llama-2-7b (in-distribution)",
            ),
            EvalSpec(
                dataset=pl.datasets.AIAuditDataset(split="test"),
                model="google/gemma-2-9b-it",
                mask=pl.masks.assistant(),
                name="Gemma-2-9b (transfer)",
            ),
            EvalSpec(
                dataset=pl.datasets.AIAuditDataset(split="test"),
                model="mistralai/Mistral-7B-Instruct-v0.2",
                mask=pl.masks.assistant(),
                name="Mistral-7B (transfer)",
            ),
        ],
    )


# Registry of standard benchmarks
_BENCHMARK_REGISTRY = {
    "deception_suite": deception_suite,
    "harmfulness_suite": harmfulness_suite,
    "cross_model_deception": cross_model_deception,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """
    Get a standard benchmark by name.

    Args:
        name: Benchmark name (see list_benchmarks() for available options)
        **kwargs: Keyword arguments passed to the benchmark factory function
                 (e.g., model="meta-llama/Llama-3-8b-Instruct")

    Returns:
        Configured Benchmark instance

    Raises:
        ValueError: If benchmark name not found

    Example:
        >>> bench = get_benchmark("deception_suite", model="meta-llama/Llama-3-8b")
        >>> results = bench.run(probe_class=pl.probes.Logistic)
    """
    if name not in _BENCHMARK_REGISTRY:
        available = ", ".join(_BENCHMARK_REGISTRY.keys())
        raise ValueError(
            f"Unknown benchmark '{name}'. Available benchmarks: {available}"
        )

    factory = _BENCHMARK_REGISTRY[name]
    return factory(**kwargs)


def list_benchmarks() -> dict[str, str]:
    """
    List all available standard benchmarks with descriptions.

    Returns:
        Dictionary mapping benchmark names to their descriptions

    Example:
        >>> benchmarks = list_benchmarks()
        >>> for name, desc in benchmarks.items():
        ...     print(f"{name}: {desc}")
    """
    benchmarks = {}
    for name, factory in _BENCHMARK_REGISTRY.items():
        # Get description from docstring
        doc = factory.__doc__ or ""
        # Extract first line of docstring
        description = doc.strip().split("\n")[0]
        benchmarks[name] = description

    return benchmarks