"""HF Trainer callbacks for reliable JORA updates."""

import os
import logging
from typing import TYPE_CHECKING, Optional

from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class JoraTrainerCallback(TrainerCallback):
    """HuggingFace Trainer callback for reliable JORA selection/temperature updates.

    This callback solves the critical issue that PyTorch's `register_full_backward_hook`
    does not trigger when module output is ModelOutput (dataclass) instead of Tensor.

    Usage:
        from peft import get_peft_model, JoraConfig
        from peft.tuners.jora.callbacks import JoraTrainerCallback

        model = get_peft_model(base_model, JoraConfig(...))
        trainer = Trainer(
            model=model,
            callbacks=[JoraTrainerCallback(model)],
            args=TrainingArguments(
                ddp_find_unused_parameters=True,  # Required for JORA
                ...
            ),
            ...
        )

    Step Semantics:
        Updates are triggered at optimizer step boundaries (after gradient accumulation),
        not per micro-batch. This differs from the original (broken) backward hook which
        would have triggered per micro-batch if it worked.

        If you need micro-batch granularity for warmup/annealing schedules, multiply
        your warmup_steps by gradient_accumulation_steps.
    """

    def __init__(self, peft_model, verbose: bool = False):
        """Initialize callback.

        Args:
            peft_model: Initial model reference. The trainer may wrap/replace this
                model later, so the callback resolves and caches the live JORA model
                from callback kwargs at train start.
            verbose: If True, log update events
        """
        self.peft_model = peft_model
        self.verbose = verbose
        self._total_steps: Optional[int] = None
        self._initialized = False
        self._jora_model = None

    def _get_jora_model(self):
        """Extract JoraModel from potentially nested PEFT wrapper."""
        from .model import JoraModel

        # Direct JoraModel
        if isinstance(self.peft_model, JoraModel):
            return self.peft_model

        # PeftModel wrapping JoraModel - check base_model attribute
        if hasattr(self.peft_model, 'base_model'):
            base = self.peft_model.base_model
            if isinstance(base, JoraModel):
                return base

        # Try to find JoraModel in module tree (depth-first search)
        def find_jora_model(module):
            if isinstance(module, JoraModel):
                return module
            for child in module.children():
                result = find_jora_model(child)
                if result is not None:
                    return result
            return None

        return find_jora_model(self.peft_model)

    def _find_jora_model_in_trainer(self, trainer_model):
        """Find JoraModel in trainer's model (may be wrapped differently)."""
        from .model import JoraModel

        if trainer_model is None:
            return None

        # Direct JoraModel
        if isinstance(trainer_model, JoraModel):
            return trainer_model

        # PeftModel wrapping JoraModel - check base_model attribute
        if hasattr(trainer_model, 'base_model'):
            base = trainer_model.base_model
            if isinstance(base, JoraModel):
                return base

        # Try to find JoraModel in module tree (depth-first search)
        def find_jora_model(module):
            if isinstance(module, JoraModel):
                return module
            for child in module.children():
                result = find_jora_model(child)
                if result is not None:
                    return result
            return None

        return find_jora_model(trainer_model)

    def _resolve_jora_model(self, trainer_model=None):
        """Resolve the live JoraModel and cache it.

        The trainer may wrap the original model after this callback is constructed
        (for example via TRL's `prepare_peft_model`). In that case, looking only at
        `self.peft_model` would miss the actual JORA instance that receives updates.
        """
        if self._jora_model is not None:
            return self._jora_model

        candidates = []
        if trainer_model is not None:
            candidates.append(trainer_model)
        if self.peft_model is not None and self.peft_model is not trainer_model:
            candidates.append(self.peft_model)

        for candidate in candidates:
            jora_model = self._find_jora_model_in_trainer(candidate)
            if jora_model is not None:
                self._jora_model = jora_model
                # Keep a reference to the trainer-owned model so later callback events
                # follow the same wrapped object used for training/saving.
                self.peft_model = candidate
                return jora_model

        return None

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState",
                       control: "TrainerControl", **kwargs):
        """Set total steps at training start."""
        trainer_model = kwargs.get("model", None)
        jora_model = self._resolve_jora_model(trainer_model)
        if jora_model is None and hasattr(state, "model"):
            jora_model = self._resolve_jora_model(state.model)

        if jora_model is None:
            logger.warning("JoraTrainerCallback: Could not find JoraModel in model hierarchy")
            return

        # Determine total steps
        if hasattr(args, 'max_steps') and args.max_steps > 0:
            self._total_steps = args.max_steps
        elif hasattr(state, 'max_steps') and state.max_steps > 0:
            self._total_steps = state.max_steps
        else:
            # Estimate from epochs
            if hasattr(state, 'num_train_epochs') and hasattr(state, 'train_dataloader'):
                try:
                    steps_per_epoch = len(state.train_dataloader) // args.gradient_accumulation_steps
                    self._total_steps = int(state.num_train_epochs * steps_per_epoch)
                except Exception:
                    pass

        if self._total_steps and hasattr(jora_model, 'set_total_steps'):
            jora_model.set_total_steps(self._total_steps)
            if self.verbose:
                logger.info(f"JoraTrainerCallback: Set total_steps={self._total_steps}")

        # Check if optimizer has JORA-specific param groups
        optimizer = kwargs.get("optimizer", None)
        if optimizer is not None:
            try:
                jora_group_names = {g.get("name", "") for g in optimizer.param_groups}
                if "jora_theta" not in jora_group_names:
                    logger.warning(
                        "JoraTrainerCallback: Optimizer does not have separate param groups "
                        "for theta/core. lr_theta and lr_core config values are being ignored. "
                        "Use model.get_optimizer_param_groups(base_lr) to create proper param groups: "
                        "groups = jora_model.get_optimizer_param_groups(base_lr=args.learning_rate); "
                        "optimizer = AdamW(groups)"
                    )
            except Exception:
                pass  # Silently ignore if optimizer doesn't have param_groups access

        self._initialized = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState",
                    control: "TrainerControl", **kwargs):
        """Trigger JORA updates at each optimizer step."""
        if not self._initialized:
            return

        jora_model = self._resolve_jora_model(kwargs.get("model", None))
        if jora_model is None:
            return

        current_step = state.global_step
        total_steps = self._total_steps or state.max_steps or 1

        # Sync step counter
        jora_model._jora_global_step = current_step

        # Check update interval
        cfg = None
        if jora_model._jora_layers:
            # Get cfg from the first adapter of the first layer
            first_layer = jora_model._jora_layers[0]
            if hasattr(first_layer, 'adapters') and first_layer.adapters:
                adapter_name = list(first_layer.adapters.keys())[0]
                cfg = first_layer.adapters[adapter_name].cfg
        update_interval = int(getattr(cfg, 'update_interval', 1)) if cfg else 1

        if update_interval <= 1 or (current_step % update_interval) == 0:
            # Update selection
            if hasattr(jora_model, 'jora_update_step'):
                jora_model.jora_update_step()
                if self.verbose:
                    logger.debug(f"JoraTrainerCallback: Updated selection at step {current_step}")

            # Update temperature
            if hasattr(jora_model, 'jora_update_temperature'):
                jora_model.jora_update_temperature(current_step, total_steps)

    def on_init_end(self, args: "TrainingArguments", state: "TrainerState",
                    control: "TrainerControl", **kwargs):
        """Called when trainer initialization ends."""
        pass

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState",
                     control: "TrainerControl", **kwargs):
        """Called when training ends."""
        pass

    def on_epoch_begin(self, args: "TrainingArguments", state: "TrainerState",
                       control: "TrainerControl", **kwargs):
        """Called when an epoch begins."""
        pass

    def on_epoch_end(self, args: "TrainingArguments", state: "TrainerState",
                     control: "TrainerControl", **kwargs):
        """Called when an epoch ends."""
        pass

    def on_step_begin(self, args: "TrainingArguments", state: "TrainerState",
                      control: "TrainerControl", **kwargs):
        """Called when a training step begins."""
        pass

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState",
                       control: "TrainerControl", **kwargs):
        """Called when a substep ends."""
        pass

    def on_log(self, args: "TrainingArguments", state: "TrainerState",
               control: "TrainerControl", **kwargs):
        """Called when logging."""
        pass

    def on_save(self, args: "TrainingArguments", state: "TrainerState",
                control: "TrainerControl", **kwargs):
        """Called when saving."""
        pass

    def on_prediction_step(self, args: "TrainingArguments", state: "TrainerState",
                          control: "TrainerControl", **kwargs):
        """Called when a prediction step is run."""
        pass


class JoraMetricsCallback(TrainerCallback):
    """Logs θ and δ parameter norms during training for diagnostic purposes.

    Tracks per-step:
      - θ_L / θ_R: mean_abs, max_abs, grad_norm (if requires_grad)
      - core δ: mean_abs, max_abs, grad_norm

    Writes diagnostics to {output_dir}/jora_diagnostics.csv for later analysis.
    """

    def __init__(self, peft_model, output_dir: str, log_interval: int = 20):
        self.peft_model = peft_model
        self.output_dir = output_dir
        self.log_interval = log_interval
        self._jora_model = None
        self._csv_path = os.path.join(output_dir, "jora_diagnostics.csv")
        self._csv_written = False
        self._header_printed = False
        # Compute θ/core LR ratio from config
        self._lr_ratio = None

    def _resolve_jora_model(self, trainer_model=None):
        from .model import JoraModel
        if isinstance(self.peft_model, JoraModel):
            return self.peft_model
        if hasattr(self.peft_model, 'base_model') and isinstance(self.peft_model.base_model, JoraModel):
            return self.peft_model.base_model

        def find_jora_model(module):
            if isinstance(module, JoraModel):
                return module
            for child in module.children():
                result = find_jora_model(child)
                if result is not None:
                    return result
            return None

        candidates = []
        if trainer_model is not None:
            candidates.append(trainer_model)
        if self.peft_model is not None and self.peft_model is not trainer_model:
            candidates.append(self.peft_model)
        for c in candidates:
            found = find_jora_model(c)
            if found is not None:
                return found
        return None

    def _collect_norms(self, jora_model):
        """Return dict of diagnostic norms across all JoraLayers."""
        stats = {
            "theta_L_mean_abs": 0.0, "theta_L_max_abs": 0.0, "theta_L_grad_norm": 0.0,
            "theta_R_mean_abs": 0.0, "theta_R_max_abs": 0.0, "theta_R_grad_norm": 0.0,
            "core_mean_abs": 0.0, "core_max_abs": 0.0, "core_grad_norm": 0.0,
            "n_layers": 0,
        }
        if not jora_model._jora_layers:
            return stats

        count = len(jora_model._jora_layers)

        for layer in jora_model._jora_layers:
            for adapter_name, adapter_state in layer.adapters.items():
                s = adapter_state

                # θ_L
                if s.theta_L is not None:
                    t = s.theta_L.detach()
                    stats["theta_L_mean_abs"] += t.abs().mean().item()
                    stats["theta_L_max_abs"] = max(stats["theta_L_max_abs"], t.abs().max().item())
                    if s.theta_L.requires_grad and s.theta_L.grad is not None:
                        stats["theta_L_grad_norm"] += s.theta_L.grad.abs().mean().item()

                # θ_R
                if s.theta_R is not None:
                    t = s.theta_R.detach()
                    stats["theta_R_mean_abs"] += t.abs().mean().item()
                    stats["theta_R_max_abs"] = max(stats["theta_R_max_abs"], t.abs().max().item())
                    if s.theta_R.requires_grad and s.theta_R.grad is not None:
                        stats["theta_R_grad_norm"] += s.theta_R.grad.abs().mean().item()

                # core delta / diag_params — handle different core types
                core_param = None
                if hasattr(s.core, 'delta') and s.core.delta is not None:
                    core_param = s.core.delta
                elif hasattr(s.core, 'diag_params') and s.core.diag_params is not None:
                    core_param = s.core.diag_params
                if core_param is not None:
                    cp_det = core_param.detach()
                    stats["core_mean_abs"] += cp_det.abs().mean().item()
                    stats["core_max_abs"] = max(stats["core_max_abs"], cp_det.abs().max().item())
                    if core_param.requires_grad and core_param.grad is not None:
                        stats["core_grad_norm"] += core_param.grad.abs().mean().item()

        stats["theta_L_mean_abs"] /= count
        stats["theta_R_mean_abs"] /= count
        stats["core_mean_abs"] /= count
        stats["n_layers"] = count
        return stats

    def _ensure_csv(self):
        if not self._csv_written:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self._csv_path, "w") as f:
                f.write("step,epoch,theta_L_mean,theta_L_max,theta_L_grad,theta_R_mean,theta_R_max,theta_R_grad,core_mean,core_max,core_grad\n")
            self._csv_written = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState",
                    control: "TrainerControl", **kwargs):
        if state.global_step % self.log_interval != 0:
            return
        jora_model = self._resolve_jora_model(kwargs.get("model", None))
        if jora_model is None:
            return

        self._ensure_csv()

        stats = self._collect_norms(jora_model)
        step = state.global_step
        epoch = round(state.epoch, 4) if state.epoch is not None else 0.0

        row = (
            f"{step},{epoch},"
            f"{stats['theta_L_mean_abs']:.6f},{stats['theta_L_max_abs']:.6f},{stats['theta_L_grad_norm']:.6f},"
            f"{stats['theta_R_mean_abs']:.6f},{stats['theta_R_max_abs']:.6f},{stats['theta_R_grad_norm']:.6f},"
            f"{stats['core_mean_abs']:.6f},{stats['core_max_abs']:.6f},{stats['core_grad_norm']:.6f}\n"
        )
        with open(self._csv_path, "a") as f:
            f.write(row)

        # Also print to trainer log for real-time monitoring
        if not self._header_printed:
            print(
                "[JoraMetrics] step | epoch | θL_mean | θL_max | θL_grad | "
                "θR_mean | θR_max | θR_grad | core_mean | core_max | core_grad"
            )
            self._header_printed = True
        if step % (self.log_interval * 5) == 0 or step <= 5:
            print(
                f"[JoraMetrics] step={step:4d} ep={epoch:.2f} | "
                f"θL={stats['theta_L_mean_abs']:.4f}/{stats['theta_L_max_abs']:.4f} g={stats['theta_L_grad_norm']:.5f} | "
                f"θR={stats['theta_R_mean_abs']:.4f}/{stats['theta_R_max_abs']:.4f} g={stats['theta_R_grad_norm']:.5f} | "
                f"δ={stats['core_mean_abs']:.4f}/{stats['core_max_abs']:.4f} g={stats['core_grad_norm']:.5f}"
            )


class JoraSchedulerCallback(TrainerCallback):
    """Optional callback for custom JORA scheduling strategies.

    This provides hooks for advanced users who want custom warmup/annealing
    behavior beyond the built-in linear schedules.
    """

    def __init__(self, peft_model,
                 selection_schedule=None,
                 temperature_schedule=None):
        """
        Args:
            peft_model: The PEFT model
            selection_schedule: Optional callable(step, total_steps) -> bool
                               Returns True if selection should update this step
            temperature_schedule: Optional callable(step, total_steps) -> float
                                  Returns temperature value for this step
        """
        self.peft_model = peft_model
        self.selection_schedule = selection_schedule
        self.temperature_schedule = temperature_schedule

    def on_step_end(self, args, state, control, **kwargs):
        from .model import JoraModel

        jora_model = None
        for module in self.peft_model.modules():
            if isinstance(module, JoraModel):
                jora_model = module
                break

        if jora_model is None:
            return

        step = state.global_step
        total = state.max_steps or 1

        # Custom selection schedule
        if self.selection_schedule and self.selection_schedule(step, total):
            jora_model.jora_update_step()

        # Custom temperature schedule
        if self.temperature_schedule:
            new_temp = self.temperature_schedule(step, total)
            for layer in jora_model._jora_layers:
                cfg = layer.cfg
                if cfg.magnitude == "oer_softmax":
                    cfg.oer_temperature = float(new_temp)
                elif cfg.magnitude == "ecd_tanh":
                    cfg.ecd_temperature = float(new_temp)
