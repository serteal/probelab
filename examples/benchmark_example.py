"""
Example demonstrating the benchmark system.

This example:
- Creates a custom benchmark that trains on DolusChat + Alpaca datasets
- Evaluates on DolusChat and CircuitBreakers test sets
- Demonstrates the train-then-eval workflow

Note: This requires GPU and will download ~8GB model + datasets.
It will take several minutes to run on a typical GPU.
"""

import probelib as pl
from probelib.benchmarks import Benchmark, EvalSpec, TrainSpec

train_eval_benchmark = Benchmark(
    name="Deception Training Demo",
    default_model="meta-llama/Llama-3.1-8B-Instruct",
    training=TrainSpec(
        datasets=[
            pl.datasets.DolusChatDataset(split="train")[:500],
            pl.datasets.AlpacaDataset(split="train")[:500],
        ],
        mask=pl.masks.assistant(),
        probe_config={
            "layer": 16,
            "sequence_aggregation": "mean",
            "C": 1.0,
        },
        train_kwargs={"batch_size": 8},
    ),
    evaluations=[
        EvalSpec(
            dataset=pl.datasets.DolusChatDataset(split="test")[:100],
            mask=pl.masks.assistant(),
            name="DolusChat Test",
        ),
        EvalSpec(
            dataset=pl.datasets.CircuitBreakersDataset(split="test")[:100],
            mask=pl.masks.assistant(),
            name="Circuit Breakers Test",
        ),
    ],
)

results = train_eval_benchmark.run(
    probe_class=pl.probes.Logistic, device="cuda", verbose=True
)
print(results.summary())
results.save("results/my_benchmark")
