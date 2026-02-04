"""Execution graph for fused multi-pipeline execution.

The ExecutionGraph builds a DAG from multiple pipelines, detecting shared
transform prefixes to avoid redundant computation. This enables efficient
batch training of multiple probes on the same activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..pipeline import Pipeline
    from ..probes.base import BaseProbe
    from ..transforms.base import ActivationTransform, ScoreTransform


@dataclass
class GraphNode:
    """A node in the execution graph.

    Each node represents either:
    - A transform step (ActivationTransform or ScoreTransform)
    - A probe step (BaseProbe)

    Attributes:
        step: The transform or probe instance
        name: Human-readable name for this node
        node_id: Unique identifier within the graph
        consumers: List of pipeline names that need this node's output
        is_probe: Whether this node is a probe (terminal for pre-transform phase)
    """

    step: "ActivationTransform | ScoreTransform | BaseProbe"
    name: str
    node_id: int
    consumers: list[str] = field(default_factory=list)
    is_probe: bool = False

    def __hash__(self) -> int:
        return self.node_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.node_id == other.node_id


@dataclass
class ExecutionGraph:
    """DAG for fused multi-pipeline execution.

    Builds a directed acyclic graph from multiple pipelines, fusing common
    prefixes to avoid redundant computation. Each unique transform configuration
    appears only once in the graph, with edges tracking data flow.

    The graph has three phases:
    1. Pre-transform phase: Shared ActivationTransforms (source → probes)
    2. Probe phase: Individual probes (no sharing, each pipeline has its own)
    3. Post-transform phase: ScoreTransforms after each probe

    Attributes:
        pipelines: Dict mapping pipeline names to Pipeline objects
        nodes: Dict mapping node IDs to GraphNode objects
        edges: Dict mapping node ID to list of child node IDs
        roots: List of root node IDs (first transforms from source)
        pipeline_paths: Dict mapping pipeline name to ordered list of node IDs
    """

    pipelines: dict[str, "Pipeline"]
    nodes: dict[int, GraphNode] = field(default_factory=dict)
    edges: dict[int, list[int]] = field(default_factory=dict)
    roots: list[int] = field(default_factory=list)
    pipeline_paths: dict[str, list[int]] = field(default_factory=dict)

    _next_id: int = field(default=0, repr=False)
    _step_to_node: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Build the execution graph from pipelines."""
        self._build_graph()

    def _get_next_id(self) -> int:
        """Get next unique node ID."""
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def _build_graph(self) -> None:
        """Build the execution graph by analyzing all pipelines.

        The algorithm:
        1. Process pipelines in order
        2. For each pipeline, walk through pre-transforms
        3. For each transform, check if an equivalent already exists WITH the same parent
        4. If yes, reuse that node; if no, create new node
        5. Track edges between consecutive transforms
        6. Create probe nodes (never shared)
        7. Create post-transform nodes (never shared)

        Note: A step can only be shared if ALL preceding steps in its path are also shared.
        This is enforced by keying on (parent_node_id, step_hash).
        """
        # Map from (parent_id, step_hash) -> node_id for prefix sharing
        # parent_id of None means it's a root node
        self._path_to_node: dict[tuple[int | None, int], int] = {}

        for pipeline_name, pipeline in self.pipelines.items():
            path: list[int] = []
            prev_node_id: int | None = None

            # Process pre-transform steps (can be shared if path matches)
            for step_name, step in pipeline._pre_steps:
                step_hash = hash(step)
                path_key = (prev_node_id, step_hash)

                # Check if equivalent step already exists with same parent
                if path_key in self._path_to_node:
                    node_id = self._path_to_node[path_key]
                    node = self.nodes[node_id]
                    # Add this pipeline as a consumer
                    if pipeline_name not in node.consumers:
                        node.consumers.append(pipeline_name)
                else:
                    # Create new node
                    node_id = self._get_next_id()
                    node = GraphNode(
                        step=step,
                        name=f"{step_name}_{node_id}",
                        node_id=node_id,
                        consumers=[pipeline_name],
                        is_probe=False,
                    )
                    self.nodes[node_id] = node
                    self.edges[node_id] = []
                    self._path_to_node[path_key] = node_id

                    # Add edge from previous node (or mark as root)
                    if prev_node_id is None:
                        if node_id not in self.roots:
                            self.roots.append(node_id)
                    else:
                        if node_id not in self.edges[prev_node_id]:
                            self.edges[prev_node_id].append(node_id)

                path.append(node_id)
                prev_node_id = node_id

            # Create probe node (never shared - each pipeline has its own)
            probe = pipeline._probe
            probe_node_id = self._get_next_id()
            probe_node = GraphNode(
                step=probe,
                name=f"{pipeline._probe_name}_{pipeline_name}",
                node_id=probe_node_id,
                consumers=[pipeline_name],
                is_probe=True,
            )
            self.nodes[probe_node_id] = probe_node
            self.edges[probe_node_id] = []

            # Edge from last pre-transform to probe (or mark as root)
            if prev_node_id is None:
                if probe_node_id not in self.roots:
                    self.roots.append(probe_node_id)
            else:
                if probe_node_id not in self.edges[prev_node_id]:
                    self.edges[prev_node_id].append(probe_node_id)

            path.append(probe_node_id)
            prev_node_id = probe_node_id

            # Process post-transform steps (not shared - tied to specific probe)
            for step_name, step in pipeline._post_steps:
                node_id = self._get_next_id()
                node = GraphNode(
                    step=step,
                    name=f"{step_name}_{pipeline_name}",
                    node_id=node_id,
                    consumers=[pipeline_name],
                    is_probe=False,
                )
                self.nodes[node_id] = node
                self.edges[node_id] = []

                # Add edge from previous node
                if prev_node_id is not None:
                    self.edges[prev_node_id].append(node_id)

                path.append(node_id)
                prev_node_id = node_id

            self.pipeline_paths[pipeline_name] = path

    def get_shared_prefix_depth(self) -> int:
        """Get the depth of shared transform prefix across all pipelines.

        Returns the number of transform nodes that are shared by ALL pipelines.
        This indicates how much computation can be reused.

        Returns:
            Number of shared prefix nodes (0 if no sharing)
        """
        if not self.pipelines:
            return 0

        # Get all paths
        paths = list(self.pipeline_paths.values())
        if not paths:
            return 0

        # Find minimum path length for pre-transforms only
        min_len = min(len(p) for p in paths)
        if min_len == 0:
            return 0

        # Find shared prefix depth (only count pre-transforms, not probes)
        shared_depth = 0
        for i in range(min_len):
            node_ids = [p[i] for p in paths]
            # Check if all paths have the same node at position i
            if len(set(node_ids)) == 1:
                node = self.nodes[node_ids[0]]
                # Stop at probe nodes
                if node.is_probe:
                    break
                shared_depth += 1
            else:
                break

        return shared_depth

    def get_execution_order(self) -> list[int]:
        """Get topological execution order for all nodes.

        Returns node IDs in an order that respects dependencies:
        a node's dependencies are always before it in the list.

        Returns:
            List of node IDs in execution order
        """
        visited: set[int] = set()
        order: list[int] = []

        def dfs(node_id: int) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for child_id in self.edges.get(node_id, []):
                dfs(child_id)
            order.append(node_id)

        for root_id in self.roots:
            dfs(root_id)

        return list(reversed(order))

    def get_pre_transform_nodes(self) -> list[GraphNode]:
        """Get all pre-transform nodes (before probes).

        Returns:
            List of GraphNodes that are ActivationTransforms
        """
        from ..transforms.base import ActivationTransform

        return [
            node
            for node in self.nodes.values()
            if isinstance(node.step, ActivationTransform)
        ]

    def get_probe_nodes(self) -> list[GraphNode]:
        """Get all probe nodes.

        Returns:
            List of GraphNodes that are probes
        """
        return [node for node in self.nodes.values() if node.is_probe]

    def get_post_transform_nodes(self) -> list[GraphNode]:
        """Get all post-transform nodes (after probes).

        Returns:
            List of GraphNodes that are ScoreTransforms
        """
        from ..transforms.base import ScoreTransform

        return [
            node
            for node in self.nodes.values()
            if isinstance(node.step, ScoreTransform)
        ]

    def summary(self) -> str:
        """Get a human-readable summary of the graph.

        Returns:
            Multi-line string describing graph structure
        """
        lines = [
            f"ExecutionGraph with {len(self.pipelines)} pipelines:",
            f"  Total nodes: {len(self.nodes)}",
            f"  Pre-transforms: {len(self.get_pre_transform_nodes())}",
            f"  Probes: {len(self.get_probe_nodes())}",
            f"  Post-transforms: {len(self.get_post_transform_nodes())}",
            f"  Shared prefix depth: {self.get_shared_prefix_depth()}",
            "",
            "Pipeline paths:",
        ]

        for name, path in self.pipeline_paths.items():
            path_str = " -> ".join(
                self.nodes[nid].name for nid in path
            )
            lines.append(f"  {name}: {path_str}")

        return "\n".join(lines)

    @classmethod
    def from_pipelines(
        cls,
        pipelines: dict[str, "Pipeline"] | list["Pipeline"],
    ) -> "ExecutionGraph":
        """Create an ExecutionGraph from pipelines.

        Args:
            pipelines: Either a dict mapping names to pipelines,
                      or a list of pipelines (auto-named pipeline_0, etc.)

        Returns:
            ExecutionGraph with fused transform structure
        """
        if isinstance(pipelines, list):
            pipelines = {f"pipeline_{i}": p for i, p in enumerate(pipelines)}

        return cls(pipelines=pipelines)
