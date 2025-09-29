"""Core Benchmark class for running evaluations."""

import logging
from datetime import datetime
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probelib.benchmarks.results import BenchmarkResults
from probelib.benchmarks.specs import EvalSpec, TrainSpec
from probelib.datasets.base import DialogueDataset
from probelib.probes.base import BaseProbe
from probelib.scripts.workflows import evaluate_probes, train_probes

logger = logging.getLogger(__name__)


class Benchmark:
    """
    A benchmark is a collection of evaluation tasks with optional training.

    Supports two modes:
    1. Evaluation-only: Pass pre-trained probe to .run()
    2. Train-then-eval: Specify training config, train from scratch

    Args:
        name: Human-readable benchmark name
        evaluations: List of EvalSpec defining what to evaluate
        training: Optional TrainSpec for training before evaluation
        default_model: Default model name/path for all specs without explicit model
        description: Optional description of the benchmark

    Example:
        >>> benchmark = Benchmark(
        ...     name="Deception Detection",
        ...     default_model="meta-llama/Llama-2-7b-chat-hf",
        ...     training=TrainSpec(
        ...         datasets=[pl.datasets.AIAuditDataset(split="train")],
        ...         mask=pl.masks.assistant(),
        ...         probe_config={"layer": 16, "sequence_aggregation": "mean"}
        ...     ),
        ...     evaluations=[
        ...         EvalSpec(
        ...             dataset=pl.datasets.AIAuditDataset(split="test"),
        ...             mask=pl.masks.assistant()
        ...         )
        ...     ]
        ... )
        >>> results = benchmark.run(probe_class=pl.probes.Logistic)
    """

    def __init__(
        self,
        name: str,
        evaluations: list[EvalSpec],
        training: TrainSpec | None = None,
        default_model: str | None = None,
        description: str = "",
    ):
        self.name = name
        self.evaluations = evaluations
        self.training = training
        self.default_model = default_model
        self.description = description

        # Validate
        if not evaluations:
            raise ValueError("Must specify at least one evaluation")

    def run(
        self,
        probe: BaseProbe | dict[str, BaseProbe] | None = None,
        probe_class: type[BaseProbe] | None = None,
        tokenizer: AutoTokenizer | None = None,
        device: str = "cuda",
        verbose: bool = True,
    ) -> BenchmarkResults:
        """
        Run the benchmark.

        Args:
            probe: Pre-trained probe(s). If None, must have training config.
            probe_class: Probe class to instantiate (if training from scratch)
            tokenizer: Tokenizer (loaded automatically if None)
            device: Device to run on
            verbose: Print progress

        Returns:
            BenchmarkResults with per-dataset metrics and trained probe

        Raises:
            ValueError: If invalid combination of arguments provided
        """
        # Validation
        if probe is None and probe_class is None:
            raise ValueError(
                "Must provide either 'probe' (pre-trained) or 'probe_class' "
                "(to train from scratch)"
            )
        if probe is not None and probe_class is not None:
            raise ValueError(
                "Cannot specify both 'probe' and 'probe_class'. "
                "Use 'probe' for pre-trained evaluation or 'probe_class' for training."
            )
        if probe is None and self.training is None:
            raise ValueError(
                "Cannot train probe without training config. "
                "Either pass a pre-trained 'probe' or add training config to benchmark."
            )

        if verbose:
            logger.info(f"Running benchmark: {self.name}")
            if self.description:
                logger.info(f"Description: {self.description}")

        # Training phase
        trained_model_name = None
        if probe is None:
            if verbose:
                logger.info("Training probe...")

            trained_model_name = self.training.model or self.default_model
            if trained_model_name is None:
                raise ValueError(
                    "No model specified for training. Set training.model or "
                    "benchmark.default_model"
                )

            # Load model and tokenizer for training
            if verbose:
                logger.info(f"Loading model for training: {trained_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                trained_model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(trained_model_name)

            # Create probe instance
            probe = probe_class(**self.training.probe_config)

            # Combine training datasets
            if len(self.training.datasets) == 1:
                train_data = self.training.datasets[0]
            else:
                # Concatenate multiple datasets
                train_data = self.training.datasets[0]
                for ds in self.training.datasets[1:]:
                    train_data = train_data + ds

            train_labels = train_data.labels

            if verbose:
                logger.info(
                    f"Training on {len(train_data)} examples from "
                    f"{len(self.training.datasets)} dataset(s)"
                )

            # Train
            train_probes(
                probe,
                model,
                tokenizer,
                train_data,
                labels=train_labels,
                mask=self.training.mask,
                device=device,
                **self.training.train_kwargs,
            )

            if verbose:
                logger.info("Training complete")

            # Clean up training model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluation phase
        results = {}
        current_model = None
        current_model_name = None

        for i, eval_spec in enumerate(self.evaluations):
            eval_model_name = eval_spec.model or self.default_model
            if eval_model_name is None:
                raise ValueError(
                    f"No model specified for evaluation '{eval_spec.name}'. "
                    f"Set eval.model or benchmark.default_model"
                )

            if verbose:
                logger.info(
                    f"Evaluation {i+1}/{len(self.evaluations)}: {eval_spec.name}"
                )

            # Load model if needed (reuse if same as previous)
            if current_model_name != eval_model_name:
                # Clean up previous model
                if current_model is not None:
                    del current_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if verbose:
                    logger.info(f"Loading model: {eval_model_name}")

                current_model = AutoModelForCausalLM.from_pretrained(
                    eval_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                )
                current_model_name = eval_model_name

                # Load tokenizer if not provided
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(eval_model_name)

            # Evaluate
            if verbose:
                logger.info(f"Evaluating on {len(eval_spec.dataset)} examples")

            _, metrics = evaluate_probes(
                probe,
                current_model,
                tokenizer,
                eval_spec.dataset,
                labels=eval_spec.dataset.labels,
                mask=eval_spec.mask,
                metrics=eval_spec.metrics,
                device=device,
            )

            results[eval_spec.name] = metrics

            if verbose:
                # Print key metrics
                if isinstance(metrics, dict):
                    key_metrics = {
                        k: v for k, v in metrics.items() if k in ["auroc", "accuracy"]
                    }
                    logger.info(f"Results: {key_metrics}")

        # Clean up final model
        if current_model is not None:
            del current_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create results object
        benchmark_results = BenchmarkResults(
            benchmark_name=self.name,
            results=results,
            probe=probe,
            config=self._get_config(),
            timestamp=datetime.now().isoformat(),
        )

        if verbose:
            logger.info("Benchmark complete!")
            logger.info(f"\nSummary:\n{benchmark_results.summary()}")

        return benchmark_results

    def _get_config(self) -> dict[str, Any]:
        """Get full benchmark configuration for reproducibility."""
        config = {
            "name": self.name,
            "description": self.description,
            "default_model": self.default_model,
            "n_evaluations": len(self.evaluations),
            "evaluation_names": [e.name for e in self.evaluations],
        }

        if self.training is not None:
            config["training"] = {
                "n_datasets": len(self.training.datasets),
                "model": self.training.model,
                "probe_config": self.training.probe_config,
                "train_kwargs": self.training.train_kwargs,
            }

        return config

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Benchmark(name='{self.name}', "
            f"n_evaluations={len(self.evaluations)}, "
            f"has_training={self.training is not None})"
        )