"""Train a probe on synthetic activations.

This example is fully self-contained (no model, no network). It mirrors the
real workflow: build ``Activations``, reduce them to per-sample features, fit a
probe, and evaluate. Swap the synthetic tensor for activations from any
collector and the probing code is unchanged.

Run with::

    python examples/train_probe_synthetic.py
"""

import torch

import probelab as pl


def main() -> None:
    torch.manual_seed(0)

    n_train, n_test = 96, 32
    seq_len, hidden_size = 24, 128
    n = n_train + n_test

    # A weak linear signal injected into class-1 samples so the probe has
    # something to learn.
    labels = torch.tensor([0, 1] * (n // 2))
    direction = torch.randn(hidden_size)
    data = torch.randn(n, seq_len, hidden_size)
    data += (labels.float()[:, None, None]) * 0.5 * direction

    train = pl.Activations(data[:n_train], dims="bsh")
    test = pl.Activations(data[n_train:], dims="bsh")

    # Feature probes train on one vector per sample.
    train_feats = train.mean("s")  # [B, H]
    test_feats = test.mean("s")

    probe = pl.probes.Logistic(seed=0).fit(train_feats, labels[:n_train])
    scores = probe.predict(test_feats)

    y_test = labels[n_train:]
    print("AUROC:        ", round(pl.metrics.auroc(y_test, scores), 3))
    print("Accuracy:     ", round(pl.metrics.accuracy(y_test, scores), 3))
    print("Recall@1%FPR: ", round(pl.metrics.recall_at_fpr(y_test, scores, fpr=0.01), 3))

    # A sequence probe learns its own pooling over the token axis instead.
    seq_probe = pl.probes.Attention(hidden_dim=32, n_epochs=50, seed=0)
    seq_probe.fit(train, labels[:n_train])
    seq_scores = seq_probe.predict(test)
    print("Attention AUROC:", round(pl.metrics.auroc(y_test, seq_scores), 3))


if __name__ == "__main__":
    main()
