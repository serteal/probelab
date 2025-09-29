"""Benchmark system for systematic evaluation of probes."""

from probelib.benchmarks.specs import EvalSpec, TrainSpec
from probelib.benchmarks.results import BenchmarkResults
from probelib.benchmarks.core import Benchmark
from probelib.benchmarks.standard import get_benchmark, list_benchmarks

__all__ = [
    "EvalSpec",
    "TrainSpec",
    "BenchmarkResults",
    "Benchmark",
    "get_benchmark",
    "list_benchmarks",
]