"""Activation collection adapters.

``collect_activations`` / ``stream_activations`` run a model over tokenized
dialogues and return :class:`~probelab.activations.Activations` (or stream
backend-neutral :class:`ActivationChunk` objects).

The functions are reachable as ``probelab.collection.collect_activations`` and
import lazily: the optional ``mirin`` backend is only imported when a function
is actually called, so ``import probelab`` stays light and dependency-free.

Example::

    import probelab as pl
    from probelab import collection

    tokens = pl.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
    acts = collection.collect_activations(model, tokens, layers=[12], pool="mean")
"""

from .mirin import collect_activations, stream_activations
from .types import ActivationChunk

__all__ = [
    "ActivationChunk",
    "collect_activations",
    "stream_activations",
]
