from __future__ import annotations

from typing import Any, Optional

import warnings

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists

from .config import JoraConfig
from .layer import JoraLayer

# Filter out the specific PyTorch warning about non-tensor module outputs
# This warning is harmless for JORA as we don't use grad_input in our hook
warnings.filterwarnings(
    "ignore",
    message="For backward hooks to be called, module output should be a Tensor or a tuple of Tensors but received .*",
    category=UserWarning
)

class JoraModel(BaseTuner):
    """PEFT BaseTuner implementation for JORA."""

    prefix = "jora_"
    tuner_layer_cls = JoraLayer

    def __init__(self, model: nn.Module, peft_config: JoraConfig, adapter_name: str = "default"):
        super().__init__(model, peft_config, adapter_name)

        # Auto-step: make JORA work with standard HF Trainer loops without manual calls.
        self._jora_global_step: int = 0
        self._jora_total_steps: int | None = None
        self._jora_layers: list[JoraLayer] = [m for m in self.model.modules() if isinstance(m, JoraLayer)]
        self._jora_hook_handle = self.model.register_full_backward_hook(self._jora_post_backward_hook)

        # If distributed is initialized, sparse selection can leave some params unused per step.
        # Users should set ddp_find_unused_parameters=True (HF TrainingArguments) to avoid DDP errors.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if getattr(peft_config, "ddp_allow_unused_parameters", True):
                warnings.warn(
                    "JORA uses sparse per-step parameter usage; for DDP set ddp_find_unused_parameters=True (TrainingArguments).",
                    RuntimeWarning,
                )

        # Ensure common transformer helper methods are available on the instance by delegating to wrapped model.
        try:
            if hasattr(self.model, "prepare_inputs_for_generation"):
                # bind underlying method
                self.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        except Exception:
            # best-effort; if model missing, leave delegation to the fallback method
            pass


    def set_total_steps(self, total_steps: int):
        """Optionally tell JORA the planned number of training steps.

        Needed only if you use warmup_ratio or temperature annealing.
        """
        self._jora_total_steps = max(0, int(total_steps))
    def _group_layers_for_selection(self):
        """Group layers for shared selection to reduce computation."""
        if not hasattr(self, '_selection_groups') or not self._selection_groups:
            cfg = self._jora_layers[0].cfg if self._jora_layers else None
            if not cfg:
                self._selection_groups = [self._jora_layers]
                return

            group_size = int(getattr(cfg, 'selection_group_size', 1))
            group_by = getattr(cfg, 'selection_group_by', 'dimension')

            if group_size <= 1:
                # Per-module selection (original behavior)
                self._selection_groups = [[layer] for layer in self._jora_layers]
            else:
                # Grouped selection for performance
                self._selection_groups = self._create_selection_groups(group_by, group_size)

    def _create_selection_groups(self, group_by: str, max_group_size: int):
        """Create groups of layers that can share selection computation."""
        if group_by == "dimension":
            # Group by input/output dimensions
            groups = {}
            for layer in self._jora_layers:
                # Get layer dimensions as grouping key
                try:
                    in_features, out_features = layer.base_layer.in_features, layer.base_layer.out_features
                    key = (in_features, out_features)
                except AttributeError:
                    # Fallback for other layer types
                    key = str(type(layer.base_layer))

                if key not in groups:
                    groups[key] = []
                groups[key].append(layer)

            # Convert to list of groups, respecting max_group_size
            result_groups = []
            for group in groups.values():
                # Split large groups if needed
                for i in range(0, len(group), max_group_size):
                    result_groups.append(group[i:i + max_group_size])

            return result_groups

        elif group_by == "type":
            # Group by module type
            groups = {}
            for layer in self._jora_layers:
                key = str(type(layer.base_layer))
                if key not in groups:
                    groups[key] = []
                groups[key].append(layer)

            # Convert to list of groups, respecting max_group_size
            result_groups = []
            for group in groups.values():
                for i in range(0, len(group), max_group_size):
                    result_groups.append(group[i:i + max_group_size])

            return result_groups

        else:
            # No grouping - fall back to per-module
            return [[layer] for layer in self._jora_layers]

    def _jora_post_backward_hook(self, module: nn.Module, grad_input, grad_output):
        # Trigger only during training.
        if not module.training:
            return

        self._jora_global_step += 1  # per backward call (micro-batch), not per optimizer step
        total_steps = self._jora_total_steps

        # Get config parameters (assume all layers use same config)
        if not self._jora_layers:
            return

        cfg = self._jora_layers[0].cfg
        update_interval = int(getattr(cfg, 'update_interval', 1))

        # Update only when needed (dramatically reduce call frequency)
        if update_interval <= 1 or (self._jora_global_step % update_interval) == 0:
            # Ensure groups are created
            self._group_layers_for_selection()

            # Batch temperature update (if needed)
            if total_steps is not None and total_steps > 0:
                for m in self._jora_layers:
                    m.update_temperature(self._jora_global_step, total_steps)

            # Pair selection update - use grouped selection for performance
            for group in self._selection_groups:
                if len(group) == 1:
                    # Single layer - use original per-module selection
                    group[0].update_step(current_step=self._jora_global_step, total_steps=total_steps)
                else:
                    # Multiple layers - use shared selection for the group
                    self._update_group_selection_shared(group, self._jora_global_step, total_steps)

    def _update_group_selection_shared(self, group, current_step: int, total_steps: int | None):
        """Update selection for a group of layers using shared computation.

        Uses the first layer as representative to compute selection, then copies
        the result to other layers in the group. This reduces computation by
        factor of group_size while maintaining reasonable selection quality.
        """
        if not group:
            return

        # Use first layer as representative for selection computation
        representative_layer = group[0]

        # Temporarily store original selection results
        original_pairs_L = representative_layer.adapters['default'].pairs_L.clone()
        original_pairs_R = representative_layer.adapters['default'].pairs_R.clone()
        original_num_pairs_L = representative_layer.adapters['default'].num_pairs_L.clone()
        original_num_pairs_R = representative_layer.adapters['default'].num_pairs_R.clone()

        # Compute selection for representative layer
        representative_layer.update_step(current_step=current_step, total_steps=total_steps)

        # Get the computed selection results
        computed_pairs_L = representative_layer.adapters['default'].pairs_L.clone()
        computed_pairs_R = representative_layer.adapters['default'].pairs_R.clone()
        computed_num_pairs_L = representative_layer.adapters['default'].num_pairs_L.clone()
        computed_num_pairs_R = representative_layer.adapters['default'].num_pairs_R.clone()

        # Apply the same selection to all other layers in the group
        for layer in group[1:]:  # Skip the representative layer
            adapter = layer.adapters['default']

            # Copy selection results (ensure shape compatibility)
            if adapter.pairs_L.shape == computed_pairs_L.shape:
                adapter.pairs_L.copy_(computed_pairs_L)
            if adapter.pairs_R.shape == computed_pairs_R.shape:
                adapter.pairs_R.copy_(computed_pairs_R)
            if adapter.num_pairs_L.shape == computed_num_pairs_L.shape:
                adapter.num_pairs_L.copy_(computed_num_pairs_L)
            if adapter.num_pairs_R.shape == computed_num_pairs_R.shape:
                adapter.num_pairs_R.copy_(computed_num_pairs_R)

            # Update Python-side cache to maintain consistency
            if hasattr(adapter, '_num_pairs_py_initialized'):
                adapter._num_pairs_py = {
                    'left': int(computed_num_pairs_L.item()),
                    'right': int(computed_num_pairs_R.item())
                }
                if hasattr(adapter, '_counter_cache'):
                    adapter._counter_cache = {
                        'left': int(computed_num_pairs_L.item()),
                        'right': int(computed_num_pairs_R.item())
                    }

    def _prepare_adapter_config(self, peft_config: JoraConfig, model_config: Optional[dict[str, Any]] = None) -> JoraConfig:
        # If user didn't pass target_modules, we keep it None and rely on explicit config.
        return peft_config

    def _check_target_module_exists(self, peft_config: JoraConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    def _create_and_replace(
        self,
        peft_config: JoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        # Replace supported modules only
        if not hasattr(target, "weight"):
            return
        new_module = JoraLayer(base_layer=target, adapter_name=adapter_name, cfg=peft_config)
        setattr(parent, target_name, new_module)

    # Optional: helpers that your trainer/callback can call
    def jora_update_step(self):
        # Use grouped selection for performance
        self._group_layers_for_selection()
        for group in self._selection_groups:
            if len(group) == 1:
                group[0].update_step(current_step=self._jora_global_step, total_steps=self._jora_total_steps)
            else:
                self._update_group_selection_shared(group, self._jora_global_step, self._jora_total_steps)

    def jora_update_temperature(self, current_step: int, total_steps: int):
        for m in self.model.modules():
            if isinstance(m, JoraLayer):
                m.update_temperature(current_step, total_steps)

    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        # Freeze all parameters first
        for _, p in model.named_parameters():
            p.requires_grad = False

        # Enable trainable flags for JORA adapter parameters only
        for m in model.modules():
            if isinstance(m, JoraLayer):
                # Adapter state lives under m.adapters[adapter_name]
                for adapter_state in m.adapters.values():
                    for param in adapter_state.parameters():
                        param.requires_grad = True

    def disable_adapter_layers(self) -> None:
        for m in self.model.modules():
            if isinstance(m, JoraLayer):
                m.enable_adapters(False)

    def enable_adapter_layers(self) -> None:
        for m in self.model.modules():
            if isinstance(m, JoraLayer):
                m.enable_adapters(True)

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        # Accept list or single adapter name
        if isinstance(adapter_name, (list, tuple)):
            if len(adapter_name) == 0:
                return
            if len(adapter_name) > 1:
                warnings.warn("Multiple active adapters not supported for JORA; using first entry")
            adapter_name = adapter_name[0]

        for module in self.model.modules():
            if isinstance(module, JoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)

    # Delegate some transformer-like methods to the wrapped model so PeftModel expectations are satisfied.
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Delegate call to the underlying base model's prepare_inputs_for_generation.
        This is required because PeftModel expects the adapter wrapper to expose this method.
        """
        if hasattr(self, "model") and hasattr(self.model, "prepare_inputs_for_generation"):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        # Some wrapped models store the transformer under `.model`
        base = getattr(self, "model", None)
        if base is not None and hasattr(base, "model") and hasattr(base.model, "prepare_inputs_for_generation"):
            return base.model.prepare_inputs_for_generation(*args, **kwargs)
        raise AttributeError("Underlying model does not implement prepare_inputs_for_generation")

