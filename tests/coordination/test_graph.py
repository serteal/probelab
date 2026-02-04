"""Tests for ExecutionGraph fusion logic."""

import pytest

from probelab import Pipeline
from probelab.transforms import pre, post
from probelab.probes import Logistic
from probelab.coordination.graph import ExecutionGraph, GraphNode


class TestTransformEquality:
    """Test that transforms can be compared for fusion."""

    def test_select_layer_equality(self):
        """Test SelectLayer equality."""
        s1 = pre.SelectLayer(16)
        s2 = pre.SelectLayer(16)
        s3 = pre.SelectLayer(20)

        assert s1 == s2
        assert s1 != s3
        assert hash(s1) == hash(s2)
        assert hash(s1) != hash(s3)

    def test_select_layers_equality(self):
        """Test SelectLayers equality."""
        s1 = pre.SelectLayers([16, 20])
        s2 = pre.SelectLayers([16, 20])
        s3 = pre.SelectLayers([16, 24])

        assert s1 == s2
        assert s1 != s3
        assert hash(s1) == hash(s2)

    def test_pool_equality(self):
        """Test Pool equality."""
        p1 = pre.Pool(dim="sequence", method="mean")
        p2 = pre.Pool(dim="sequence", method="mean")
        p3 = pre.Pool(dim="sequence", method="max")
        p4 = pre.Pool(dim="layer", method="mean")

        assert p1 == p2
        assert p1 != p3
        assert p1 != p4
        assert hash(p1) == hash(p2)

    def test_post_pool_equality(self):
        """Test post.Pool equality."""
        p1 = post.Pool(method="mean")
        p2 = post.Pool(method="mean")
        p3 = post.Pool(method="max")

        assert p1 == p2
        assert p1 != p3
        assert hash(p1) == hash(p2)


class TestExecutionGraphBuild:
    """Test ExecutionGraph construction."""

    def test_single_pipeline(self):
        """Test graph with single pipeline."""
        pipeline = Pipeline([
            ("select", pre.SelectLayer(16)),
            ("pool", pre.Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        graph = ExecutionGraph.from_pipelines({"test": pipeline})

        assert len(graph.pipelines) == 1
        assert len(graph.nodes) == 3  # select, pool, probe
        assert len(graph.roots) == 1
        assert "test" in graph.pipeline_paths
        assert len(graph.pipeline_paths["test"]) == 3

    def test_identical_prefixes_are_fused(self):
        """Test that identical transform prefixes are shared."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)

        # Should have: 1 shared SelectLayer + 1 shared Pool + 2 separate probes = 4 nodes
        assert len(graph.nodes) == 4

        # First two nodes in each path should be the same
        assert graph.pipeline_paths["p1"][0] == graph.pipeline_paths["p2"][0]  # SelectLayer
        assert graph.pipeline_paths["p1"][1] == graph.pipeline_paths["p2"][1]  # Pool
        assert graph.pipeline_paths["p1"][2] != graph.pipeline_paths["p2"][2]  # Probes differ

    def test_partial_prefix_sharing(self):
        """Test that only matching prefix is shared."""
        pipelines = {
            "mean_pool": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "max_pool": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="max")),
                ("probe", Logistic()),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)

        # Should have: 1 shared SelectLayer + 2 different Pools + 2 probes = 5 nodes
        assert len(graph.nodes) == 5

        # First node should be shared (SelectLayer)
        assert graph.pipeline_paths["mean_pool"][0] == graph.pipeline_paths["max_pool"][0]
        # Pool nodes should differ
        assert graph.pipeline_paths["mean_pool"][1] != graph.pipeline_paths["max_pool"][1]

    def test_no_sharing_with_different_layers(self):
        """Test that different layers are not shared."""
        pipelines = {
            "layer_16": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "layer_20": Pipeline([
                ("select", pre.SelectLayer(20)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)

        # Should have: 2 SelectLayers + 2 Pools + 2 probes = 6 nodes
        # (Pool nodes are separate because they follow different SelectLayers)
        assert len(graph.nodes) == 6

        # No sharing expected
        assert graph.pipeline_paths["layer_16"][0] != graph.pipeline_paths["layer_20"][0]

    def test_post_transforms_not_shared(self):
        """Test that post-probe transforms are not shared."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("probe", Logistic()),
                ("pool", post.Pool(method="mean")),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("probe", Logistic()),
                ("pool", post.Pool(method="mean")),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)

        # Should have: 1 shared SelectLayer + 2 probes + 2 post pools = 5 nodes
        assert len(graph.nodes) == 5
        assert len(graph.get_post_transform_nodes()) == 2


class TestExecutionGraphMethods:
    """Test ExecutionGraph utility methods."""

    def test_get_shared_prefix_depth(self):
        """Test shared prefix depth calculation."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)
        assert graph.get_shared_prefix_depth() == 2  # SelectLayer + Pool

    def test_shared_prefix_depth_partial(self):
        """Test partial prefix sharing depth."""
        pipelines = {
            "mean": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "max": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("pool", pre.Pool(dim="sequence", method="max")),
                ("probe", Logistic()),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)
        assert graph.get_shared_prefix_depth() == 1  # Only SelectLayer

    def test_shared_prefix_depth_no_sharing(self):
        """Test when no sharing is possible."""
        pipelines = {
            "l16": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("probe", Logistic()),
            ]),
            "l20": Pipeline([
                ("select", pre.SelectLayer(20)),
                ("probe", Logistic()),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)
        assert graph.get_shared_prefix_depth() == 0

    def test_get_execution_order(self):
        """Test topological execution order."""
        pipeline = Pipeline([
            ("select", pre.SelectLayer(16)),
            ("pool", pre.Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        graph = ExecutionGraph.from_pipelines({"test": pipeline})
        order = graph.get_execution_order()

        assert len(order) == 3
        # Each node should come before its children
        for i, node_id in enumerate(order):
            children = graph.edges.get(node_id, [])
            for child_id in children:
                assert order.index(child_id) > i

    def test_get_node_types(self):
        """Test filtering nodes by type."""
        pipeline = Pipeline([
            ("select", pre.SelectLayer(16)),
            ("pool", pre.Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
            ("post_pool", post.Pool(method="mean")),
        ])

        graph = ExecutionGraph.from_pipelines({"test": pipeline})

        pre_nodes = graph.get_pre_transform_nodes()
        probe_nodes = graph.get_probe_nodes()
        post_nodes = graph.get_post_transform_nodes()

        assert len(pre_nodes) == 2  # SelectLayer, Pool
        assert len(probe_nodes) == 1  # Logistic
        assert len(post_nodes) == 1  # post.Pool

    def test_summary(self):
        """Test summary string generation."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(16)),
                ("probe", Logistic()),
            ]),
        }

        graph = ExecutionGraph.from_pipelines(pipelines)
        summary = graph.summary()

        assert "ExecutionGraph" in summary
        assert "2 pipelines" in summary
        assert "p1" in summary
        assert "p2" in summary


class TestExecutionGraphFromList:
    """Test creating ExecutionGraph from list of pipelines."""

    def test_from_list_auto_names(self):
        """Test that list pipelines get auto-named."""
        pipelines = [
            Pipeline([
                ("select", pre.SelectLayer(16)),
                ("probe", Logistic()),
            ]),
            Pipeline([
                ("select", pre.SelectLayer(20)),
                ("probe", Logistic()),
            ]),
        ]

        graph = ExecutionGraph.from_pipelines(pipelines)

        assert "pipeline_0" in graph.pipelines
        assert "pipeline_1" in graph.pipelines
