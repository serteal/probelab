"""Tests for vmapped ensemble training utilities."""

import torch
import torch.nn as nn

from probelab.utils.vmap_ensemble import VmapEnsemble, gated_bipolar_regularization


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def test_vmap_ensemble_trains_tracks_best_and_loads_states():
    torch.manual_seed(0)
    nets = [TinyLinear(), TinyLinear()]
    ensemble = VmapEnsemble(
        nets,
        learning_rates=[0.05, 0.1],
        weight_decays=[0.0, 0.01],
        max_epochs=[2, 1],
        patience=2,
        device="cpu",
        dtype=torch.float32,
        n_forward_args=1,
    )
    x = torch.randn(5, 3)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    before = ensemble.eval_forward(x).detach().clone()
    ensemble.train_step(x, y)
    after = ensemble.eval_forward(x).detach()

    assert after.shape == (2, 5)
    assert not torch.allclose(before, after)

    ensemble.check_val(after, y)
    assert torch.isfinite(ensemble.best_val_loss).all()

    ensemble.mark_epoch_done(0)
    assert ensemble.n_active == 1

    state = ensemble.extract_state(0)
    assert set(state) == {"linear.weight", "linear.bias"}

    probes = [TinyLinear(), TinyLinear()]
    ensemble.load_into_probes(probes)
    loaded_logits = probes[0](x)
    expected_logits = torch.func.functional_call(
        ensemble.meta_model,
        ({k: v[0] for k, v in ensemble.best_params.items()}, {}),
        (x,),
    )
    torch.testing.assert_close(loaded_logits, expected_logits)


def test_gated_bipolar_regularization_returns_per_probe_penalties():
    params = {
        "W_proj.weight": torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ])
    }

    penalties = gated_bipolar_regularization(
        params,
        gate_dim=2,
        lambda_l1=0.1,
        lambda_orth=0.2,
    )

    assert penalties.shape == (2,)
    assert torch.isfinite(penalties).all()
    assert (penalties >= 0).all()
    assert penalties[1] > penalties[0]
