from typing import TYPE_CHECKING, Any, Callable, Self

import torch

from ..types import HookPoint
from .architectures import get_arch

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class HookedModel:
    """Context manager for extracting activations from specific model layers.

    This class implements several optimizations for efficient activation extraction:
    - Single shared hook function to reduce overhead
    - Optional inference mode for faster forward passes
    - Buffer reuse to minimize memory allocations
    - Non-blocking GPU transfers
    - Model layer truncation to save memory
    - Position-aware extraction for memory efficiency

    Usage:
        with HookedModel(model, layers=[10, 20]) as hooked:
            activations = hooked.get_activations(inputs)
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        layers: list[int],
        detach_activations: bool = False,
        hook_point: HookPoint = HookPoint.POST_BLOCK,
        target_device: str | torch.device | None = None,
    ):
        self.model = model
        self.layers = layers
        self.layer_set = set(layers)
        self.cache = {}
        self.hooks = []
        self.architecture = get_arch(model)
        self.original_layers = None
        self.detach_activations = detach_activations
        self.hook_point = hook_point
        self.target_device = torch.device(target_device) if isinstance(target_device, str) else target_device
        self._resolved_device: torch.device | None = None

        # For PEFT models, we need to work with the base model for layer access
        self.base_model = model
        if hasattr(model, "get_base_model"):
            self.base_model = model.get_base_model()  # type: ignore

    def _resolve_target_device(self) -> torch.device:
        """Determine the device to consolidate activations onto.

        - If ``target_device`` was set explicitly, use it.
        - If all cached layers live on the same device, use that (single-GPU path).
        - Otherwise fall back to CPU to avoid slow GPU-to-GPU transfers.
        """
        if self.target_device is not None:
            return self.target_device
        devices = {self.cache[layer].device for layer in self.layers}
        if len(devices) == 1:
            return devices.pop()  # single GPU — backward compatible
        return torch.device("cpu")  # multi GPU — skip GPU↔GPU transfers

    def _create_shared_hook(self) -> Callable:
        """Create a single shared hook function for all layers.

        - For hook_point=HookPoint.PRE_LAYERNORM, the hook is attached to the input_layernorm
          and captures its output (legacy behavior).
        - For hook_point=HookPoint.POST_BLOCK, the hook is attached to the transformer block
          and captures the block's output (aligns with HF hidden_states for that layer).
        """

        def shared_hook(module, input, output):  # type: ignore
            # Get the layer index from module metadata
            layer_idx = getattr(module, "_hook_layer_idx", None)
            if layer_idx is None or layer_idx not in self.layer_set:
                return

            # Some modules return tuples; first element is hidden_states
            out = output[0] if isinstance(output, (tuple, list)) else output
            if self.detach_activations:
                out = out.detach()
            self.cache[layer_idx] = out

        return shared_hook

    def __enter__(self) -> Self:
        """Register hooks and truncate model layers."""
        max_layer = max(self.layers)

        # Create a single shared hook function
        shared_hook = self._create_shared_hook()

        # Store original layers for restoration
        self.original_layers = self.architecture.get_layers(self.base_model)

        # Register hooks according to hook_point
        for layer in self.layers:
            if self.hook_point == HookPoint.PRE_LAYERNORM:
                module = self.architecture.get_layer_norm(self.base_model, layer)
            else:  # HookPoint.POST_BLOCK
                module = self.architecture.get_layer_module(self.base_model, layer)

            # Attach layer index as metadata to the module
            setattr(module, "_hook_layer_idx", layer)
            # Use the same shared hook function for all layers
            self.hooks.append(module.register_forward_hook(shared_hook))

        # Truncate layers to save memory
        self.architecture.set_layers(
            self.base_model, self.original_layers[: max_layer + 1]
        )

        return self

    def get_activations(
        self,
        batch_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get activations for a batch of inputs."""
        # Use inference_mode for better performance when not needing gradients
        context = (
            torch.inference_mode() if self.detach_activations else torch.enable_grad()
        )
        with context:
            model_inputs = {
                k: v for k, v in batch_inputs.items() if k != "detection_mask"
            }
            # Activation extraction does not reuse KV cache across calls.
            # Disabling it avoids unnecessary allocation/work in CausalLM wrappers.
            model_inputs.setdefault("use_cache", False)
            try:
                _ = self.model(**model_inputs)  # type: ignore
            except TypeError as exc:
                # Fallback for architectures that do not accept use_cache.
                if "use_cache" not in str(exc):
                    raise
                model_inputs.pop("use_cache", None)
                _ = self.model(**model_inputs)  # type: ignore
            # Normalize devices — with device_map="auto" layers may live on
            # different GPUs.  Lazily resolve the target once so we avoid
            # slow GPU↔GPU transfers (consolidate on CPU when multi-GPU).
            if self._resolved_device is None:
                self._resolved_device = self._resolve_target_device()
            dev = self._resolved_device
            is_cpu = dev.type == "cpu"
            moved = [
                self.cache[layer].to(dev, non_blocking=not is_cpu)
                for layer in self.layers
            ]
            result = torch.stack(moved, dim=0)
            # Clear cache to free GPU memory immediately
            self.cache.clear()
            return result

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Restore model and remove hooks."""
        # Restore original layers
        if self.original_layers is not None:
            self.architecture.set_layers(self.base_model, self.original_layers)

        # Remove hooks
        for hook in self.hooks:
            hook.remove()

        # Clear cache to free any remaining references
        self.cache.clear()
