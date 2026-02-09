"""Tests for lazy import behavior in package init and dataset registry."""

import json
import subprocess
import sys
import textwrap


def _run_python_snippet(code: str) -> dict:
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        text=True,
    ).strip()
    return json.loads(out)


def test_import_probelab_exposes_top_level_api():
    """`import probelab` should expose the expected top-level API."""
    result = _run_python_snippet(textwrap.dedent("""
        import json
        import sys
        import probelab

        print(json.dumps({
            "has_activations": hasattr(probelab, "Activations"),
            "has_collect": hasattr(probelab, "collect_activations"),
            "has_tokenize_dataset": hasattr(probelab, "tokenize_dataset"),
            "has_datasets": hasattr(probelab, "datasets"),
            "has_metrics": hasattr(probelab, "metrics"),
        }))
    """))

    assert result["has_activations"] is True
    assert result["has_collect"] is True
    assert result["has_tokenize_dataset"] is True
    assert result["has_datasets"] is True
    assert result["has_metrics"] is True


def test_dataset_registry_initializes_on_first_use():
    """Dataset loader modules should register only when dataset APIs are used."""
    result = _run_python_snippet(textwrap.dedent("""
        import json
        import probelab.datasets as datasets_mod
        import probelab.datasets.registry as registry

        before = {
            "initialized": registry._REGISTRY_INITIALIZED,
            "size": len(registry.REGISTRY),
        }
        names = datasets_mod.list_datasets()
        after = {
            "initialized": registry._REGISTRY_INITIALIZED,
            "size": len(registry.REGISTRY),
            "count": len(names),
        }
        print(json.dumps({"before": before, "after": after}))
    """))

    assert result["before"]["initialized"] is False
    assert result["before"]["size"] == 0
    assert result["after"]["initialized"] is True
    assert result["after"]["size"] > 0
    assert result["after"]["count"] > 0
