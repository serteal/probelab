"""Temporary probe API rewrite snapshot.

Runs every probe on deterministic synthetic activations and records predictions
plus learned-tensor digests. Intended for before/after refactors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

import probelab as pl


def _make_acts(n_samples: int = 12, seq: int = 6, d_model: int = 8):
    torch.manual_seed(1234)
    half = n_samples // 2
    t = torch.randn(n_samples, seq, d_model) * 0.05
    t[:half, :, 0] += 1.5
    t[half:, :, 0] -= 1.5
    # Vary sequence masks so sequence batching/padding is exercised.
    det = torch.ones(n_samples, seq, dtype=torch.bool)
    det[1, -2:] = False
    det[4, -1:] = False
    det[8, -3:] = False
    labels = torch.tensor([1] * half + [0] * half)
    return pl.Activations.from_padded(t, det, dims="bsh"), labels


def _tensor_digest(t: torch.Tensor) -> dict[str, Any]:
    v = t.detach().cpu().float().reshape(-1)
    if v.numel() == 0:
        return {"shape": list(t.shape), "sum": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "shape": list(t.shape),
        "sum": float(v.sum().item()),
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()),
    }


def _probe_tensors(probe) -> dict[str, dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    if isinstance(probe, torch.nn.Module):
        for name, tensor in probe.state_dict().items():
            tensors[name] = tensor
    for attr in ("scaler_mean", "scaler_std", "global_mean"):
        tensor = getattr(probe, attr, None)
        if isinstance(tensor, torch.Tensor):
            tensors[attr] = tensor
    return {name: _tensor_digest(tensor) for name, tensor in sorted(tensors.items())}


def _construct_probes() -> dict[str, tuple[object, str]]:
    common = {"device": "cpu", "seed": 123}
    return {
        "logistic": (pl.probes.Logistic(C=0.7, max_iter=10, **common), "feature"),
        "mlp": (pl.probes.MLP(hidden_dim=6, dropout=0.0, n_epochs=3, batch_size=4, **common), "feature"),
        "mass_mean": (pl.probes.MassMean(**common), "feature"),
        "bilinear": (pl.probes.Bilinear(rank=4, n_epochs=3, batch_size=4, **common), "feature"),
        "tpc": (pl.probes.TPC(max_degree=3, rank=4, n_epochs=2, batch_size=4, **common), "feature"),
        "ee_mlp": (pl.probes.EEMLP(n_layers=2, hidden_dim=6, dropout=0.0, n_epochs=3, batch_size=4, **common), "feature"),
        "attention": (pl.probes.Attention(hidden_dim=6, dropout=0.0, n_epochs=3, patience=10, batch_size=4, val_split=0.25, **common), "sequence"),
        "multimax": (pl.probes.MultiMax(n_heads=3, mlp_hidden_dim=6, dropout=0.0, n_epochs=3, patience=10, batch_size=4, val_split=0.25, **common), "sequence"),
        "gated_bipolar": (pl.probes.GatedBipolar(mlp_hidden_dim=6, gate_dim=4, dropout=0.0, n_epochs=3, patience=10, batch_size=4, val_split=0.25, **common), "sequence"),
        "positional_attention": (pl.probes.PositionalAttention(n_heads=3, n_epochs=3, patience=10, batch_size=4, val_split=0.25, **common), "sequence"),
        "soft_attention": (pl.probes.SoftAttention(n_heads=3, hidden_dim=6, dropout=0.0, n_epochs=3, patience=10, batch_size=4, val_split=0.25, **common), "sequence"),
        "mha": (pl.probes.MHA(proj_dim=8, n_heads=2, n_enc_layers=1, dropout=0.0, n_epochs=3, patience=10, batch_size=4, val_split=0.25, **common), "sequence"),
        "rolling_attention": (pl.probes.RollingAttention(n_heads=3, hidden_dim=6, window_size=3, dropout=0.0, n_epochs=3, patience=10, batch_size=4, val_split=0.25, **common), "sequence"),
    }


def run_snapshot() -> dict[str, Any]:
    torch.use_deterministic_algorithms(False)
    acts, labels = _make_acts()
    pooled = acts.mean("s")
    out: dict[str, Any] = {}
    for name, (probe, kind) in _construct_probes().items():
        torch.manual_seed(123)
        train_x = acts if kind == "sequence" else pooled
        probe.fit(train_x, labels)
        probs = probe.predict(train_x).detach().cpu().float()
        out[name] = {
            "shape": list(probs.shape),
            "predictions": probs.reshape(-1).tolist(),
            "pred_digest": _tensor_digest(probs),
            "state": _probe_tensors(probe),
        }
    return out


def _assert_close(current: dict[str, Any], baseline: dict[str, Any], tol: float) -> None:
    def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
        return {name.removeprefix("net."): value for name, value in state.items()}

    failures: list[str] = []
    for name in sorted(baseline):
        if name not in current:
            failures.append(f"{name}: missing current result")
            continue
        b = torch.tensor(baseline[name]["predictions"], dtype=torch.float32)
        c = torch.tensor(current[name]["predictions"], dtype=torch.float32)
        if b.shape != c.shape:
            failures.append(f"{name}: prediction shape {tuple(c.shape)} != {tuple(b.shape)}")
            continue
        max_abs = float((b - c).abs().max().item()) if b.numel() else 0.0
        if not math.isfinite(max_abs) or max_abs > tol:
            failures.append(f"{name}: max prediction delta {max_abs:.8g} > {tol}")
        baseline_state = normalize_state(baseline[name]["state"])
        current_state = normalize_state(current[name]["state"])
        if set(baseline_state) != set(current_state):
            missing = sorted(set(baseline_state) - set(current_state))
            extra = sorted(set(current_state) - set(baseline_state))
            failures.append(f"{name}: state keys differ; missing={missing}, extra={extra}")
            continue
        for state_name in sorted(baseline_state):
            b_state = baseline_state[state_name]
            c_state = current_state[state_name]
            if b_state["shape"] != c_state["shape"]:
                failures.append(f"{name}.{state_name}: shape {c_state['shape']} != {b_state['shape']}")
                continue
            for field in ("sum", "mean", "std"):
                delta = abs(float(b_state[field]) - float(c_state[field]))
                if not math.isfinite(delta) or delta > tol:
                    failures.append(f"{name}.{state_name}: {field} delta {delta:.8g} > {tol}")
                    break
    if failures:
        raise SystemExit("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "compare"])
    parser.add_argument("--path", default="/tmp/probelab-api-rewrite/probe_snapshot.json")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    snapshot = run_snapshot()
    path = Path(args.path)
    if args.mode == "write":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        print(f"wrote {path}")
    else:
        baseline = json.loads(path.read_text())
        _assert_close(snapshot, baseline, args.tol)
        print(f"matched {path} within {args.tol}")


if __name__ == "__main__":
    main()
