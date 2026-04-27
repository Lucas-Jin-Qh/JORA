from __future__ import annotations

from typing import Any, Optional

import warnings

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils import AuxiliaryTrainingWrapper

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
    """PEFT BaseTuner implementation for JORA.

    Important: Pair selection updates require JoraTrainerCallback (recommended) or
    manual calls to jora_model.jora_update_step() after each optimizer step.
    The deprecated _jora_post_backward_hook method is no longer used.
    """

    prefix = "jora_"
    tuner_layer_cls = JoraLayer

    def __init__(self, model: nn.Module, peft_config: JoraConfig, adapter_name: str = "default"):
        super().__init__(model, peft_config, adapter_name)

        # Callback-driven: Use JoraTrainerCallback for reliable updates instead of broken backward hooks.
        # The original backward hook approach failed because PyTorch hooks don't trigger on ModelOutput.
        self._jora_global_step: int = 0
        self._jora_total_steps: int | None = None
        self._jora_layers: list[JoraLayer] = [m for m in self.model.modules() if isinstance(m, JoraLayer)]

        # Lazy initialization of selection groups
        self._selection_groups = None

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
            cfg = None
            if self._jora_layers:
                # Get cfg from the first adapter of the first layer
                first_layer = self._jora_layers[0]
                if hasattr(first_layer, 'adapters') and first_layer.adapters:
                    adapter_name = list(first_layer.adapters.keys())[0]
                    cfg = first_layer.adapters[adapter_name].cfg

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
        """DEPRECATED: This method is no longer used for pair selection updates.

        Pair selection is now driven by JoraTrainerCallback (recommended) or manual
        calls to jora_update_step(). This backward hook is kept for reference only
        and will be removed in a future version.

        The original backward hook approach failed because PyTorch hooks don't trigger
        on ModelOutput objects, which are commonly used in transformer models.
        """
        warnings.warn(
            "_jora_post_backward_hook is deprecated and not used for updates. "
            "Use JoraTrainerCallback or manually call jora_update_step() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Return immediately - do not execute old logic
        return

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

        # Compute selection for representative layer
        representative_layer.update_step(current_step=current_step, total_steps=total_steps)
        rep_adapter_name = representative_layer.active_adapter
        rep_adapter = representative_layer.adapters[rep_adapter_name]

        # Get the computed selection results
        computed_pairs_L = rep_adapter.pairs_L.clone()
        computed_pairs_R = rep_adapter.pairs_R.clone()
        computed_num_pairs_L = rep_adapter.num_pairs_L.clone()
        computed_num_pairs_R = rep_adapter.num_pairs_R.clone()
        computed_pairs_frozen = getattr(rep_adapter, '_pairs_frozen', False)
        computed_step_idx = rep_adapter.step_idx.clone()

        # Apply the same selection to all other layers in the group
        for layer in group[1:]:  # Skip the representative layer
            adapter_name = layer.active_adapter
            adapter = layer.adapters[adapter_name]

            # Copy selection results (ensure shape compatibility)
            if adapter.pairs_L.shape == computed_pairs_L.shape:
                adapter.pairs_L.copy_(computed_pairs_L)
            if adapter.pairs_R.shape == computed_pairs_R.shape:
                adapter.pairs_R.copy_(computed_pairs_R)
            if adapter.num_pairs_L.shape == computed_num_pairs_L.shape:
                adapter.num_pairs_L.copy_(computed_num_pairs_L)
            if adapter.num_pairs_R.shape == computed_num_pairs_R.shape:
                adapter.num_pairs_R.copy_(computed_num_pairs_R)
            if adapter.step_idx.shape == computed_step_idx.shape:
                adapter.step_idx.copy_(computed_step_idx)

            adapter._pairs_frozen = computed_pairs_frozen
            adapter._step_idx_py = int(computed_step_idx.item())

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
        # Paper path: selective_diag requires square layers (in_features == out_features).
        # Rectangular MLP projections (e.g. gate_proj, up_proj, down_proj) are not yet
        # supported by the paper-exact operator; skip them to avoid silent corruption.
        if peft_config.core == "selective_diag" and hasattr(target, "in_features") and hasattr(target, "out_features"):
            if target.in_features != target.out_features:
                warnings.warn(
                    f"JORA paper path (core='selective_diag') requires square layers but '{current_key}' "
                    f"has in_features={target.in_features} != out_features={target.out_features}. "
                    f"Skipping this module. To include rectangular layers, use a non-selective_diag core.",
                    UserWarning,
                    stacklevel=4,
                )
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

        # Enable trainable flags for JORA adapter parameters and modules_to_save
        for m in model.modules():
            if isinstance(m, JoraLayer):
                # Adapter state lives under m.adapters[adapter_name]
                for adapter_state in m.adapters.values():
                    for param in adapter_state.parameters():
                        param.requires_grad = True
            elif isinstance(m, AuxiliaryTrainingWrapper):
                # Enable trainable flags for modules_to_save (e.g., lm_head)
                for param in m.parameters():
                    param.requires_grad = True

    def get_optimizer_param_groups(self, base_lr: float = 1e-4) -> list[dict]:
        """Return optimizer param groups with JORA-specific LR scaling.

        JORA has two parameter types with different gradient scales:
        - theta (rotation angles): high leverage, needs lower LR
        - core/delta (diagonal scaling): directly scales activations

        This method creates separate param groups so they can be trained
        with different learning rates. Without this, lr_theta and lr_core
        config values are ignored and all params share base_lr.

        Usage:
            model = get_peft_model(base_model, jora_config)
            groups = model.get_optimizer_param_groups(base_lr=args.learning_rate)
            optimizer = AdamW(groups)

        Args:
            base_lr: Base learning rate for core parameters.
                    theta_lr = base_lr * (lr_theta / lr_core)

        Returns:
            List of param groups dicts with 'params', 'lr', 'name' keys.
        """
        from .layer import JoraLayer

        theta_params = []
        core_params = []
        magnitude_params = []

        for layer in self._jora_layers:
            for adapter_name, adapter_state in layer.adapters.items():
                cfg = adapter_state.cfg

                # theta_L and theta_R: rotation angles
                if adapter_state.theta_L is not None:
                    theta_params.append(adapter_state.theta_L)
                if adapter_state.theta_R is not None:
                    theta_params.append(adapter_state.theta_R)

                # core parameters (delta for selective_diag, etc.)
                for p in adapter_state.core.parameters():
                    core_params.append(p)

                # magnitude (OER logits if enabled)
                if adapter_state.ecd_log_mag is not None:
                    magnitude_params.append(adapter_state.ecd_log_mag)

        groups = []

        # theta: rotation angles — high leverage, needs lower LR
        if theta_params:
            # Use lr_theta/lr_core ratio from config to scale theta LR
            lr_core_val = float(getattr(cfg, 'lr_core', 1.0))
            lr_ratio = float(getattr(cfg, 'lr_theta', 1.0)) / lr_core_val if lr_core_val != 0 else 0.0
            theta_lr = base_lr * lr_ratio
            groups.append({
                "params": theta_params,
                "lr": theta_lr,
                "name": "jora_theta",
            })

        # core (delta): diagonal scaling — directly scales activations
        if core_params:
            groups.append({
                "params": core_params,
                "lr": base_lr,
                "name": "jora_core",
            })

        # magnitude: OER logits (if enabled)
        if magnitude_params:
            magnitude_lr_scale = float(getattr(cfg, 'magnitude_lr_scale', 1.0))
            groups.append({
                "params": magnitude_params,
                "lr": base_lr * magnitude_lr_scale,
                "name": "jora_magnitude",
            })

        return groups

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

