"""Results container for benchmark runs."""

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from probelib.probes.base import BaseProbe


@dataclass
class BenchmarkResults:
    """
    Results from running a benchmark.

    Args:
        benchmark_name: Name of the benchmark
        results: Nested dict mapping eval_name -> metric_name -> value
        probe: Trained probe(s) used for evaluation
        config: Full benchmark configuration for reproducibility
        timestamp: ISO format timestamp of when benchmark was run

    Example:
        >>> results = BenchmarkResults(
        ...     benchmark_name="Deception Suite",
        ...     results={
        ...         "AI Audit": {"auroc": 0.85, "accuracy": 0.78},
        ...         "AI Liar": {"auroc": 0.82, "accuracy": 0.75}
        ...     },
        ...     probe=probe,
        ...     config={...},
        ...     timestamp="2024-01-15T10:30:00"
        ... )
        >>> print(results.summary())
    """

    benchmark_name: str
    results: dict[str, dict[str, float]]
    probe: BaseProbe | dict[str, BaseProbe] | None
    config: dict[str, Any]
    timestamp: str

    def summary(self, round_digits: int = 4) -> pd.DataFrame:
        """
        Return results as a pandas DataFrame.

        Args:
            round_digits: Number of decimal places to round to

        Returns:
            DataFrame with eval names as rows and metrics as columns
        """
        df = pd.DataFrame(self.results).T
        return df.round(round_digits)

    def save(self, path: str | Path) -> None:
        """
        Save results, probe, and config to disk.

        Creates a directory with:
        - results.json: Metrics and config
        - probe.pkl: Serialized probe(s)
        - summary.csv: Results DataFrame

        Args:
            path: Directory path to save to (will be created if doesn't exist)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save results and config as JSON
        results_data = {
            "benchmark_name": self.benchmark_name,
            "results": self.results,
            "config": self.config,
            "timestamp": self.timestamp,
        }
        with open(path / "results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save probe(s)
        if self.probe is not None:
            with open(path / "probe.pkl", "wb") as f:
                pickle.dump(self.probe, f)

        # Save summary DataFrame
        summary_df = self.summary()
        summary_df.to_csv(path / "summary.csv")

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResults":
        """
        Load saved benchmark results from disk.

        Args:
            path: Directory path containing saved results

        Returns:
            Loaded BenchmarkResults instance

        Raises:
            FileNotFoundError: If results.json not found
        """
        path = Path(path)

        # Load results and config
        with open(path / "results.json", "r") as f:
            data = json.load(f)

        # Load probe if exists
        probe = None
        probe_path = path / "probe.pkl"
        if probe_path.exists():
            with open(probe_path, "rb") as f:
                probe = pickle.load(f)

        return cls(
            benchmark_name=data["benchmark_name"],
            results=data["results"],
            probe=probe,
            config=data["config"],
            timestamp=data["timestamp"],
        )

    def compare(
        self, other: "BenchmarkResults", metric: str | None = None
    ) -> pd.DataFrame:
        """
        Compare results with another benchmark run.

        Args:
            other: Another BenchmarkResults to compare with
            metric: Specific metric to compare (if None, compares all common metrics)

        Returns:
            DataFrame showing side-by-side comparison with difference column

        Example:
            >>> comparison = results1.compare(results2, metric="auroc")
            >>> print(comparison)
        """
        df1 = self.summary()
        df2 = other.summary()

        # Filter to common eval names
        common_evals = df1.index.intersection(df2.index)

        if metric is not None:
            # Compare single metric
            if metric not in df1.columns or metric not in df2.columns:
                raise ValueError(
                    f"Metric '{metric}' not found in both results. "
                    f"Available in first: {list(df1.columns)}, "
                    f"Available in second: {list(df2.columns)}"
                )

            comparison = pd.DataFrame(
                {
                    f"{self.benchmark_name} ({self.timestamp[:10]})": df1.loc[
                        common_evals, metric
                    ],
                    f"{other.benchmark_name} ({other.timestamp[:10]})": df2.loc[
                        common_evals, metric
                    ],
                }
            )
            comparison["Difference"] = (
                comparison.iloc[:, 1] - comparison.iloc[:, 0]
            )
        else:
            # Compare all common metrics
            common_metrics = df1.columns.intersection(df2.columns)

            # Create multi-level columns
            comparison = pd.DataFrame(index=common_evals)
            for m in common_metrics:
                comparison[(m, self.timestamp[:10])] = df1.loc[common_evals, m]
                comparison[(m, other.timestamp[:10])] = df2.loc[common_evals, m]
                comparison[(m, "Diff")] = (
                    df2.loc[common_evals, m] - df1.loc[common_evals, m]
                )

            comparison.columns = pd.MultiIndex.from_tuples(comparison.columns)

        return comparison

    def __repr__(self) -> str:
        """String representation showing benchmark name and number of evaluations."""
        n_evals = len(self.results)
        return (
            f"BenchmarkResults(benchmark='{self.benchmark_name}', "
            f"n_evaluations={n_evals}, timestamp='{self.timestamp}')"
        )