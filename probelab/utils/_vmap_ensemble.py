"""Vmapped ensemble training for identical-architecture networks.

Trains N networks with the same architecture but different hyperparameters
(learning rate, weight decay, max epochs) in parallel using torch.func.vmap.
Includes a vectorized AdamW optimizer that supports per-network LR and WD.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, stack_module_state, vmap


class VmapEnsemble:
    """Train N identical-architecture networks in parallel via vmap.

    All networks must share the same parameter shapes (same architecture
    hyperparameters like hidden_dim, n_heads, etc.). Per-network learning
    rate, weight decay, and epoch budget are supported via vectorized AdamW.

    Usage::

        nets = [MyNetwork(d_model=256, hidden=64) for _ in range(42)]
        ensemble = VmapEnsemble(
            nets,
            learning_rates=[1e-3]*21 + [1e-4]*21,
            weight_decays=[1e-3]*42,
            max_epochs=[5]*21 + [10]*21,
        )
        for epoch in range(ensemble.global_max_epochs):
            if ensemble.all_done:
                break
            for x, mask, y in batches:
                # Signature is train_step(x, y, *extra); the mask is an extra
                # forward arg, so it goes after the labels.
                ensemble.train_step(x, y, mask)
            val_logits = ensemble.eval_forward_batched(X, val_idx, batch_size)
            ensemble.check_val(val_logits, y_val)
            ensemble.mark_epoch_done(epoch)
        ensemble.restore_best()
        state_i = ensemble.extract_state(i)
    """

    def __init__(
        self,
        nets: list[nn.Module],
        learning_rates: list[float] | Tensor,
        weight_decays: list[float] | Tensor,
        max_epochs: list[int] | Tensor,
        patience: int = 5,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        returns_tuple: bool = False,
        regularization_fn: Callable[[dict[str, Tensor]], Tensor] | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        max_chunk_size: int | None = None,
        eval_only: bool = False,
        n_forward_args: int = 2,
    ):
        self.N = len(nets)
        self.max_chunk_size = max_chunk_size
        self.device = device
        self.dtype = dtype
        self.returns_tuple = returns_tuple
        self.regularization_fn = regularization_fn
        self.patience = patience
        self.betas = betas
        self.eps = eps
        self.n_forward_args = n_forward_args

        # -- Stack network parameters --
        for net in nets:
            net.train()
        params, buffers = stack_module_state(nets)
        self.params: dict[str, Tensor] = {
            k: v.detach().to(device).requires_grad_(not eval_only)
            for k, v in params.items()
        }
        self.buffers: dict[str, Tensor] = {
            k: v.detach().to(device) for k, v in buffers.items()
        }

        # -- Meta model for functional_call --
        self.meta_model = copy.deepcopy(nets[0]).to("meta")
        self.meta_model.train()

        if eval_only:
            # Forward-only mode: skip optimizer & early-stopping state
            self.lr = None
            self.wd = None
            self.max_epochs = None
            self.global_max_epochs = 0
            self.exp_avg = {}
            self.exp_avg_sq = {}
            self.step_count = 0
            self.active_mask = torch.ones(self.N, dtype=torch.bool, device=device)
            self.best_val_loss = None
            self.patience_counter = None
            self.best_params = {}
            self._vmapped_train = None
            self._vmapped_eval = self._make_vmapped_fwd(randomness="error")
            self._logged_chunk = False
            return

        # -- Per-probe hyperparameters --
        self.lr = _to_tensor(learning_rates, device, torch.float32)
        self.wd = _to_tensor(weight_decays, device, torch.float32)
        self.max_epochs = _to_tensor(max_epochs, device, torch.long)
        self.global_max_epochs: int = int(self.max_epochs.max().item())

        # -- Vectorized AdamW state (fp32 for stability) --
        self.exp_avg: dict[str, Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.params.items()
        }
        self.exp_avg_sq: dict[str, Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.params.items()
        }
        self.step_count: int = 0

        # -- Early stopping state --
        self.active_mask = torch.ones(self.N, dtype=torch.bool, device=device)
        self.best_val_loss = torch.full((self.N,), float("inf"), device=device)
        self.patience_counter = torch.zeros(self.N, dtype=torch.long, device=device)
        self.best_params: dict[str, Tensor] = {
            k: v.detach().clone() for k, v in self.params.items()
        }

        # -- Cache vmapped forward functions --
        self._vmapped_train = self._make_vmapped_fwd(randomness="different")
        self._vmapped_eval = self._make_vmapped_fwd(randomness="error")
        self._logged_chunk = False  # one-time chunk logging

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _make_vmapped_fwd(self, randomness: str):
        """Build a cached vmapped forward function."""
        meta = self.meta_model
        returns_tuple = self.returns_tuple

        if self.n_forward_args == 1:
            def fwd(params, buffers, x):
                out = functional_call(meta, (params, buffers), (x,))
                return out[0] if returns_tuple else out
            return vmap(fwd, in_dims=(0, 0, None), randomness=randomness)
        else:
            def fwd(params, buffers, x, mask):
                out = functional_call(meta, (params, buffers), (x, mask))
                return out[0] if returns_tuple else out
            return vmap(fwd, in_dims=(0, 0, None, None), randomness=randomness)

    def _estimate_chunk_size(self, x: Tensor) -> int:
        """Estimate safe number of probes to vmap in one chunk.

        Uses actually-available GPU memory (total - allocated) with conservative
        safety margins. Calls empty_cache() first to reclaim freed memory.
        """
        if self.max_chunk_size is not None:
            return min(self.max_chunk_size, self.N)
        if self.device == "cpu" or not torch.cuda.is_available():
            return self.N

        if x.dim() == 3:
            B, S, H = x.shape
        else:
            B, H = x.shape
            S = 1
        elem = 2 if self.dtype in (torch.bfloat16, torch.float16) else 4
        # Peak memory per probe during vmap forward+backward:
        # ~4 [B,S,H]-sized tensors (input, layernorm output, hidden, grad)
        bytes_per_probe = B * S * H * elem * 4

        # Reclaim any cached-but-freed memory before measuring
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        available = total - allocated
        usable = int(available * 0.7)  # 30% headroom
        return min(max(1, usable // max(1, bytes_per_probe)), self.N)

    def _train_forward(self, x: Tensor, *extra: Tensor) -> Tensor:
        """Vmapped forward in train mode. Returns [N, B] logits."""
        self.meta_model.train()
        return self._vmapped_train(self.params, self.buffers, x, *extra)

    def eval_forward(self, x: Tensor, *extra: Tensor) -> Tensor:
        """Vmapped forward in eval mode with auto-chunking. Returns [N, B] logits."""
        self.meta_model.eval()
        chunk = self._estimate_chunk_size(x)
        if chunk >= self.N:
            logits = self._vmapped_eval(self.params, self.buffers, x, *extra)
        else:
            parts: list[Tensor] = []
            for ci in range(0, self.N, chunk):
                cj = min(ci + chunk, self.N)
                p_c = {k: v[ci:cj] for k, v in self.params.items()}
                b_c = {k: v[ci:cj] for k, v in self.buffers.items()}
                parts.append(self._vmapped_eval(p_c, b_c, x, *extra))
                del p_c, b_c
            logits = torch.cat(parts, dim=0)
        self.meta_model.train()
        return logits

    def eval_forward_batched(
        self,
        X,
        indices: list[int],
        batch_size: int,
    ) -> Tensor:
        """Eval forward over minibatches. Returns [N, n_samples] logits."""
        max_batch_tokens = batch_size * 256
        all_logits: list[Tensor] = []
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            # Pre-compute max seq length to decide sub-batching BEFORE
            # materializing the padded tensor (avoids OOM on long seqs).
            local_max = max(
                int(X.offsets[j + 1]) - int(X.offsets[j]) for j in batch_idx
            )
            B_actual = len(batch_idx)
            if B_actual * local_max > max_batch_tokens and B_actual > 1:
                sub_bs = max(1, max_batch_tokens // local_max)
                for sb_start in range(0, len(batch_idx), sub_bs):
                    sb_idx = batch_idx[sb_start:sb_start + sub_bs]
                    sb_seq, sb_mask = X.pad_batch(sb_idx)
                    sb_seq = sb_seq.to(self.device, dtype=self.dtype)
                    sb_mask = sb_mask.to(self.device)
                    all_logits.append(self.eval_forward(sb_seq, sb_mask))
                    del sb_seq, sb_mask
            else:
                batch_seq, batch_mask = X.pad_batch(batch_idx)
                batch_seq = batch_seq.to(self.device, dtype=self.dtype)
                batch_mask = batch_mask.to(self.device)
                all_logits.append(self.eval_forward(batch_seq, batch_mask))
                del batch_seq, batch_mask
        return torch.cat(all_logits, dim=1)  # [N, total_samples]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, x: Tensor, y: Tensor, *extra: Tensor) -> None:
        """One vmapped forward + backward + optimizer step.

        Auto-chunks probes to fit GPU memory: each chunk's forward+backward
        runs independently so intermediates are freed between chunks.
        Gradients accumulate on the shared params via tensor views.

        Args:
            x: Input tensor — [B, S, H] (seq) or [B, H] (flat).
            y: [B] binary labels (already on device, float).
            *extra: Additional forward args (e.g. mask [B, S] for seq probes).
        """
        # Zero gradients
        for v in self.params.values():
            if v.grad is not None:
                v.grad = None

        chunk = self._estimate_chunk_size(x)

        if not self._logged_chunk and chunk < self.N:
            logger = logging.getLogger(__name__)
            shape_str = "x".join(str(d) for d in x.shape)
            alloc_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                f"      [vmap-chunk] N={self.N} chunk={chunk} "
                f"(shape={shape_str} alloc={alloc_gb:.1f}GB)"
            )
            self._logged_chunk = True

        if chunk >= self.N:
            # Fast path: all probes in one vmap call (no chunking overhead)
            logits = self._train_forward(x, *extra)  # [N, B]
            targets = y.unsqueeze(0).expand(self.N, -1)  # [N, B]
            per_probe_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ).mean(dim=1)  # [N]
            if self.regularization_fn is not None:
                per_probe_loss = per_probe_loss + self.regularization_fn(self.params)
            masked_loss = (per_probe_loss * self.active_mask.float()).sum()
            masked_loss.backward()
        else:
            # Chunked path: forward+backward per chunk to limit peak memory.
            targets = y.unsqueeze(0)  # [1, B]
            self.meta_model.train()
            for ci in range(0, self.N, chunk):
                cj = min(ci + chunk, self.N)
                if not self.active_mask[ci:cj].any():
                    continue
                n_c = cj - ci
                p_c = {k: v[ci:cj] for k, v in self.params.items()}
                b_c = {k: v[ci:cj] for k, v in self.buffers.items()}
                logits = self._vmapped_train(p_c, b_c, x, *extra)  # [n_c, B]
                c_targets = targets.expand(n_c, -1)
                ppl = F.binary_cross_entropy_with_logits(
                    logits, c_targets, reduction="none"
                ).mean(dim=1)  # [n_c]
                if self.regularization_fn is not None:
                    ppl = ppl + self.regularization_fn(p_c)
                loss = (ppl * self.active_mask[ci:cj].float()).sum()
                loss.backward()
                del logits, ppl, loss, p_c, b_c

        # Vectorized AdamW step
        self._adamw_step()

    def train_step_sequential(self, x: Tensor, y: Tensor, *extra: Tensor) -> None:
        """Sequential forward+backward per active probe (no vmap).

        Use when B*S is large enough that individual matmuls saturate GPU
        bandwidth, making vmap overhead counterproductive.
        """
        for v in self.params.values():
            if v.grad is not None:
                v.grad = None

        self.meta_model.train()
        for i in range(self.N):
            if not self.active_mask[i]:
                continue
            p_i = {k: v[i] for k, v in self.params.items()}
            b_i = {k: v[i] for k, v in self.buffers.items()}
            out = functional_call(self.meta_model, (p_i, b_i), (x, *extra))
            logits = out[0] if self.returns_tuple else out  # [B]
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if self.regularization_fn is not None:
                p_1 = {k: v[i:i+1] for k, v in self.params.items()}
                loss = loss + self.regularization_fn(p_1).squeeze(0)
            loss.backward()
            del out, logits, loss

        self._adamw_step()

    def _adamw_step(self) -> None:
        """Vectorized AdamW with per-probe LR and WD."""
        self.step_count += 1
        beta1, beta2 = self.betas
        bc1 = 1 - beta1 ** self.step_count
        bc2 = 1 - beta2 ** self.step_count

        for k in self.params:
            p = self.params[k]
            if p.grad is None:
                continue

            g = p.grad.float()  # fp32 for stability
            p_f = p.float()

            # Broadcast [N] → [N, 1, 1, ...] to match param dims
            shape = (-1,) + (1,) * (p.dim() - 1)
            lr = self.lr.view(shape)
            wd = self.wd.view(shape)
            active = self.active_mask.view(shape).float()

            # EMA of gradient and squared gradient
            self.exp_avg[k].lerp_(g, 1 - beta1)
            self.exp_avg_sq[k].lerp_(g * g, 1 - beta2)

            # Bias-corrected estimates
            m_hat = self.exp_avg[k] / bc1
            v_hat = self.exp_avg_sq[k] / bc2

            # Decoupled weight decay + Adam update, masked by active
            update = active * lr * (m_hat / (v_hat.sqrt() + self.eps) + wd * p_f)

            with torch.no_grad():
                p.copy_((p_f - update).to(self.dtype))

    # ------------------------------------------------------------------
    # Validation & early stopping
    # ------------------------------------------------------------------

    def check_val(self, val_logits: Tensor, val_y: Tensor) -> None:
        """Update per-probe early stopping from validation logits.

        Args:
            val_logits: [N, n_val] logits from eval_forward_batched.
            val_y: [n_val] binary labels.
        """
        targets = val_y.unsqueeze(0).expand(self.N, -1)
        per_probe_loss = F.binary_cross_entropy_with_logits(
            val_logits, targets, reduction="none"
        ).mean(dim=1)  # [N]

        improved = (per_probe_loss < self.best_val_loss) & self.active_mask

        # Snapshot best params for improved probes
        if improved.any():
            for k in self.params:
                mask = improved.view(-1, *([1] * (self.params[k].dim() - 1)))
                self.best_params[k] = torch.where(
                    mask, self.params[k].detach().clone(), self.best_params[k]
                )

        self.best_val_loss = torch.where(improved, per_probe_loss, self.best_val_loss)
        self.patience_counter = torch.where(
            improved,
            torch.zeros_like(self.patience_counter),
            self.patience_counter + 1,
        )
        self.active_mask &= self.patience_counter < self.patience

    def mark_epoch_done(self, epoch: int) -> None:
        """Deactivate probes that have reached their max_epochs."""
        self.active_mask &= epoch < (self.max_epochs - 1)

    def restore_best(self) -> None:
        """Copy best checkpoint params back into self.params."""
        for k in self.params:
            self.params[k].data.copy_(self.best_params[k])

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def extract_state(self, idx: int) -> dict[str, Tensor]:
        """Get a single probe's state_dict from the best checkpoint."""
        return {k: v[idx].clone() for k, v in self.best_params.items()}

    def load_into_probes(self, probes: list) -> None:
        """Write best checkpoint weights back into probe objects.

        Args:
            probes: List of probe objects (same length and order as nets
                passed to constructor). Each must expose ``load_state_dict``.
        """
        self.restore_best()
        for i, probe in enumerate(probes):
            probe.load_state_dict(self.extract_state(i))
            probe.eval()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_done(self) -> bool:
        """True when no probes are still training."""
        return not self.active_mask.any().item()

    @property
    def n_active(self) -> int:
        """Number of probes still training."""
        return int(self.active_mask.sum().item())


# ----------------------------------------------------------------------
# Regularization functions
# ----------------------------------------------------------------------


def gated_bipolar_regularization(
    params: dict[str, Tensor],
    gate_dim: int,
    lambda_l1: float = 1e-5,
    lambda_orth: float = 1e-4,
) -> Tensor:
    """Compute per-probe GatedBipolar regularization from stacked params.

    Args:
        params: Stacked parameter dict from VmapEnsemble.
        gate_dim: Gate dimension of the GatedBipolar network.
        lambda_l1: L1 sparsity coefficient.
        lambda_orth: Orthogonality penalty coefficient.

    Returns:
        [N] tensor of regularization losses.
    """
    W = params["W_proj.weight"]  # [N, gate_dim, mlp_hidden_dim]
    l1 = lambda_l1 * W.abs().sum(dim=(1, 2))  # [N]
    WtW = torch.bmm(W, W.transpose(1, 2))  # [N, gate_dim, gate_dim]
    identity = torch.eye(gate_dim, device=W.device, dtype=W.dtype).unsqueeze(0)
    orth = lambda_orth * (WtW - identity).pow(2).sum(dim=(1, 2))  # [N]
    return l1 + orth


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_tensor(
    values: list | Tensor, device: str, dtype: torch.dtype
) -> Tensor:
    """Convert list or tensor to a 1-D tensor on device."""
    if isinstance(values, Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.tensor(values, device=device, dtype=dtype)
