"""Coordination module for multi-pipeline execution with step fusion."""

from .graph import ExecutionGraph
from .pipeline_set import PipelineSet

__all__ = ["ExecutionGraph", "PipelineSet"]
