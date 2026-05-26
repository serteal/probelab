"""Base classes and shared training helpers for probes."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..activations import Activations
from ..batching import iter_feature_batches, iter_sequence_batches


class BaseProbe(nn.Module):
    """Base class for estimator-style probes that are also ``nn.Module``s.

    High-level methods ``fit`` and ``predict`` operate on :class:`Activations`.
    Lower-level methods such as ``initialize``, ``forward``, ``loss_on_batch``,
    and ``predict_tensor`` operate on tensors and are intended for custom
    PyTorch training loops.
    """

    _CAST_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        device: str | None = None,
        seed: int | None = None,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        cast: str | None = None,
    ):
        super().__init__()
        self.device = device
        self.seed = seed
        self._optimizer_fn = optimizer_fn
        self._scheduler_fn = scheduler_fn
        self.cast = cast
        self._training_dtype: torch.dtype | None = None
        self._initialized = False
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_state: dict[str, torch.Tensor] | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def fitted(self) -> bool:
        return self.initialized

    def _mark_initialized(self, dtype: torch.dtype) -> None:
        self._initialized = True
        self._training_dtype = dtype

    def _check_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} must be initialized before use")

    def _resolve_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        if self.cast is not None:
            return self._CAST_MAP[self.cast]
        return input_dtype

    def _module_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        for tensor in list(self.parameters()) + list(self.buffers()):
            if tensor.numel() > 0:
                return tensor.device, tensor.dtype
        device = torch.device(self.device or "cpu")
        return device, self._training_dtype or torch.float32

    def _seed_everything(self) -> torch.Generator | None:
        """Return a deterministic local generator without mutating caller RNG state."""
        return self._make_generator()

    @contextmanager
    def _temporary_seed(self):
        """Temporarily seed RNGs for deterministic init/training, then restore them."""
        if self.seed is None:
            yield
            return

        py_state = random.getstate()
        np_state = np.random.get_state()
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            try:
                yield
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)

    def _make_generator(self) -> torch.Generator | None:
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

    def _make_optimizer(self, params, **defaults):
        if self._optimizer_fn is not None:
            return self._optimizer_fn(params)
        fused = isinstance(self.device, str) and self.device.startswith("cuda")
        return AdamW(params, fused=fused, **defaults)

    def _make_scheduler(self, optimizer):
        if self._scheduler_fn is not None:
            return self._scheduler_fn(optimizer)
        return None

    def _reset_best_tracking(self) -> None:
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_state = None

    def _step_scheduler(self, scheduler, val_loss: float | None = None) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, ReduceLROnPlateau):
            if val_loss is not None:
                scheduler.step(val_loss)
            return
        scheduler.step()

    def _to_labels(self, y) -> torch.Tensor:
        if isinstance(y, torch.Tensor):
            return y
        return torch.tensor([label.value if hasattr(label, "value") else label for label in y])

    def _fit_kwargs(self, **kwargs):
        return {
            "batch_size": kwargs.get("batch_size", self.batch_size),
            "max_padded_tokens": kwargs.get("max_padded_tokens", getattr(self, "max_padded_tokens", None)),
            "sort_by_length": kwargs.get("sort_by_length", True),
            "shuffle": kwargs.get("shuffle", True),
            "val_split": kwargs.get("val_split", getattr(self, "val_split", 0.2)),
            "eval_interval": kwargs.get("eval_interval", getattr(self, "eval_interval", 1)),
            "grad_accum_steps": kwargs.get("grad_accum_steps", 1),
            "device": kwargs.get("device", self.device),
        }

    def _reject_layer_axis(self, X: Activations) -> None:
        if "l" in X.dims:
            raise ValueError(
                f"{self.__class__.__name__} expects no LAYER axis. "
                f'Call select("l", layer) first. Current dims: {X.dims}'
            )

    def _require_sequence_activations(self, X: Activations) -> None:
        self._reject_layer_axis(X)
        if "s" not in X.dims:
            raise ValueError(f"{self.__class__.__name__} probe requires SEQ axis")

    def _feature_data_from_activations(
        self, X: Activations, y: list | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._reject_layer_axis(X)
        labels = None if y is None else self._to_labels(y)
        if "s" in X.dims:
            features, tokens_per_sample = X.extract_tokens()
            if labels is not None:
                if labels.ndim == 1:
                    labels = torch.repeat_interleave(labels, tokens_per_sample.to(labels.device))
                elif labels.ndim == 2:
                    _, det_bool = X.to_padded()
                    det_flat = det_bool.reshape(-1).to(labels.device)
                    label_flat = labels.reshape(-1)
                    labels = label_flat[: det_flat.numel()][det_flat[: label_flat.numel()]]
                else:
                    raise ValueError(f"Invalid label shape: {tuple(labels.shape)}")
        else:
            features = X.data
        return features, labels

    def _feature_predict_from_flat(self, X: Activations, flat_probs: torch.Tensor) -> torch.Tensor:
        if "s" not in X.dims:
            return flat_probs
        _, padded_mask = X.to_padded()
        probs = torch.zeros_like(padded_mask, dtype=flat_probs.dtype, device=flat_probs.device)
        probs[padded_mask.to(flat_probs.device)] = flat_probs
        return probs

    def regularization_loss(self) -> torch.Tensor:
        device, dtype = self._module_device_dtype()
        return torch.zeros((), device=device, dtype=dtype)

    def loss_on_batch(self, *tensors: torch.Tensor) -> torch.Tensor:
        *inputs, labels = tensors
        logits = self(*inputs)
        return F.binary_cross_entropy_with_logits(logits, labels) + self.regularization_loss()

    def predict_tensor(self, *tensors: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(*tensors))

    def configure_optimizer(self, params=None):
        return self._make_optimizer(
            self.parameters() if params is None else params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _snapshot_best(self) -> None:
        self._best_state = copy.deepcopy(self.state_dict())

    def _check_val_loss(self, val_loss: float, scheduler=None) -> bool:
        # ``scheduler`` is retained for internal compatibility; training loops
        # step schedulers on their epoch cadence, not validation cadence.
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            self._snapshot_best()
        else:
            self._patience_counter += 1
        return self._patience_counter >= self.patience

    def _restore_best(self) -> None:
        if self._best_state is not None:
            self.load_state_dict(self._best_state)

    def _save_probe(self, path: Path | str, init_kwargs: dict) -> None:
        self._check_initialized()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class": self.__class__.__name__,
                "init_kwargs": init_kwargs,
                "seed": self.seed,
                "device": self.device,
                "cast": self.cast,
                "training_dtype": str(self._training_dtype),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def _stored_dtype(cls, state: dict) -> torch.dtype:
        dtype_str = state["training_dtype"]
        return getattr(torch, dtype_str.split(".")[-1])

    # ------------------------------------------------------------------
    # Default high-level fit loops used by concrete probes.
    # ------------------------------------------------------------------

    def _fit_feature_default(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        dataloader: bool = False,
        shuffle_with_generator: bool = True,
        **kwargs,
    ):
        opts = self._fit_kwargs(**kwargs)
        labels = labels.to(dtype=self._training_dtype)
        g = self._make_generator() if shuffle_with_generator else None
        optimizer = self.configure_optimizer()
        scheduler = self._make_scheduler(optimizer)
        self._reset_best_tracking()
        with self._temporary_seed():
            self.train()
            for _ in range(self.n_epochs):
                device, dtype = self._module_device_dtype()
                if dataloader:
                    dataset = torch.utils.data.TensorDataset(features, labels)
                    batches = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=opts["batch_size"],
                        shuffle=opts["shuffle"],
                        generator=g,
                    )
                else:
                    batches = (
                        (batch_features, batch_labels)
                        for batch_features, batch_labels, _ in iter_feature_batches(
                            features,
                            labels,
                            batch_size=opts["batch_size"],
                            shuffle=opts["shuffle"],
                            generator=g,
                        )
                    )
                for batch_features, batch_labels in batches:
                    batch_features = batch_features.to(device=device, dtype=dtype)
                    batch_labels = batch_labels.to(device=device, dtype=dtype)
                    optimizer.zero_grad()
                    loss = self.loss_on_batch(batch_features, batch_labels)
                    loss.backward()
                    optimizer.step()
                self._step_scheduler(scheduler)
        self.eval()
        return self

    def _fit_sequence_default(self, X: Activations, labels: torch.Tensor, **kwargs):
        opts = self._fit_kwargs(**kwargs)
        if "s" not in X.dims:
            raise ValueError(f"{self.__class__.__name__} probe requires SEQ axis")
        labels = labels.to(opts["device"] or self.device or X.data.device, dtype=self._training_dtype)

        g = self._make_generator()
        n_samples = X.batch_size
        n_val = max(1, int(opts["val_split"] * n_samples))
        indices = torch.randperm(n_samples, device="cpu", generator=g)
        train_idx, val_idx = indices[n_val:].tolist(), indices[:n_val].tolist()
        batch_size = min(opts["batch_size"], len(train_idx)) if train_idx else opts["batch_size"]
        val_y = labels[val_idx]

        optimizer = self.configure_optimizer()
        scheduler = self._make_scheduler(optimizer)
        self._reset_best_tracking()

        device, dtype = self._module_device_dtype()
        with self._temporary_seed():
            self.train()
            for epoch in range(self.n_epochs):
                step_in_accum = 0
                optimizer.zero_grad()
                for batch_seq, batch_mask, batch_labels, _ in iter_sequence_batches(
                    X.data,
                    X.offsets,
                    X.detection_mask,
                    labels,
                    indices=train_idx,
                    batch_size=batch_size,
                    max_padded_tokens=opts["max_padded_tokens"],
                    sort_by_length=opts["sort_by_length"],
                    shuffle=opts["shuffle"],
                    generator=g,
                ):
                    batch_seq = batch_seq.to(device=device, dtype=dtype)
                    batch_mask = batch_mask.to(device=device)
                    batch_labels = batch_labels.to(device=device, dtype=dtype)
                    loss = self.loss_on_batch(batch_seq, batch_mask, batch_labels)
                    loss = loss / opts["grad_accum_steps"]
                    loss.backward()
                    step_in_accum += 1
                    if step_in_accum % opts["grad_accum_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                if step_in_accum % opts["grad_accum_steps"] != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                val_loss = None
                should_stop = False
                if epoch % opts["eval_interval"] == 0:
                    self.eval()
                    with torch.no_grad():
                        val_logits = self._sequence_logits(
                            X,
                            val_idx,
                            batch_size=batch_size,
                            max_padded_tokens=opts["max_padded_tokens"],
                            sort_by_length=True,
                        )
                        val_loss = F.binary_cross_entropy_with_logits(
                            val_logits, val_y.to(val_logits.device, dtype=val_logits.dtype)
                        ).item()
                    should_stop = self._check_val_loss(val_loss)
                    if not should_stop:
                        self.train()
                self._step_scheduler(scheduler, val_loss)
                if should_stop:
                    break

        self._restore_best()
        self.eval()
        return self

    def _sequence_logits(
        self,
        X: Activations,
        indices: list[int],
        *,
        batch_size: int,
        max_padded_tokens: int | None = None,
        sort_by_length: bool = False,
    ) -> torch.Tensor:
        device, dtype = self._module_device_dtype()
        if not indices:
            return torch.empty(0, device=device, dtype=dtype)
        logits_by_position = torch.empty(len(indices), device=device, dtype=dtype)
        filled = torch.zeros(len(indices), dtype=torch.bool)
        positions = {int(sample_idx): pos for pos, sample_idx in enumerate(indices)}
        for batch_seq, batch_mask, _, batch_idx in iter_sequence_batches(
            X.data,
            X.offsets,
            X.detection_mask,
            None,
            indices=indices,
            batch_size=batch_size,
            max_padded_tokens=max_padded_tokens,
            sort_by_length=sort_by_length,
            shuffle=False,
        ):
            batch_seq = batch_seq.to(device=device, dtype=dtype)
            batch_mask = batch_mask.to(device=device)
            batch_logits = self(batch_seq, batch_mask)
            for local_pos, sample_idx in enumerate(batch_idx.detach().cpu().tolist()):
                pos = positions[int(sample_idx)]
                logits_by_position[pos] = batch_logits[local_pos]
                filled[pos] = True
        if not filled.all():
            raise RuntimeError("sequence batching did not return logits for every requested index")
        return logits_by_position
