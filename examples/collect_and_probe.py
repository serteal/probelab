"""End-to-end: dataset -> tokenize -> collect activations -> probe.

Unlike ``train_probe_synthetic.py`` this requires a real model and the
collection extra::

    pip install "probelab[collection]"
    # plus the mirin backend from https://github.com/serteal/mirin

It is written to be read as a template; edit ``MODEL_NAME`` / ``LAYER`` and run::

    python examples/collect_and_probe.py
"""

from __future__ import annotations

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 12


def main() -> None:
    import mirin  # noqa: F401  (clear error if the backend is missing)
    from transformers import AutoTokenizer

    import probelab as pl
    from probelab import collection

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = mirin.Model.from_pretrained(MODEL_NAME)

    # 1. Data: combine a positive and a negative source into one labelled set.
    dataset = pl.datasets.load("circuit_breakers")
    train, test = dataset.split(0.8, stratified=True, seed=0)

    # 2. Tokenize, scoring only assistant tokens. Pass template= if your model
    #    is a renamed/fine-tuned checkpoint that auto-detection misses.
    mask = pl.masks.assistant()
    train_tokens = pl.tokenize_dataset(train, tokenizer, mask=mask)
    test_tokens = pl.tokenize_dataset(test, tokenizer, mask=mask)

    # 3. Collect pooled (mean) activations for a single layer.
    train_acts = collection.collect_activations(
        model, train_tokens, layers=LAYER, pool="mean", progress=True
    )
    test_acts = collection.collect_activations(
        model, test_tokens, layers=LAYER, pool="mean", progress=True
    )

    # 4. Train and evaluate.
    probe = pl.probes.Logistic(seed=0).fit(train_acts, train.labels)
    scores = probe.predict(test_acts)
    print("AUROC:", round(pl.metrics.auroc(test.labels, scores), 3))

    # Optionally persist the probe and the activations.
    probe.save("circuit_breakers_logistic.pt")
    pl.storage.save(test_acts, "test_acts.h5", dtype="bfloat16")


if __name__ == "__main__":
    main()
