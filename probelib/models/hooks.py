from typing import TYPE_CHECKING, Any, Callable, Self

import torch

from ..types import HookPoint
from .architectures import ArchitectureRegistry

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
    ):
        self.model = model
        self.layers = layers
        self.layer_set = set(layers)
        self.cache = {}
        self.hooks = []
        self.architecture = ArchitectureRegistry.get_architecture(model)
        self.original_layers = None
        self.detach_activations = detach_activations
        self.hook_point = hook_point

        # For PEFT models, we need to work with the base model for layer access
        self.base_model = model
        if hasattr(model, "get_base_model"):
            self.base_model = model.get_base_model()  # type: ignore

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
            _ = self.model(**model_inputs)  # type: ignore
            result = torch.stack([self.cache[layer] for layer in self.layers], dim=0)
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

        # Clear GPU cache when exiting context
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
