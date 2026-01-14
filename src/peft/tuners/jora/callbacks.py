"""HF Trainer callbacks for reliable JORA updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class JoraTrainerCallback:
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
            peft_model: The PEFT-wrapped model containing JoraModel
            verbose: If True, log update events
        """
        self.peft_model = peft_model
        self.verbose = verbose
        self._total_steps: Optional[int] = None
        self._initialized = False

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

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState",
                       control: "TrainerControl", **kwargs):
        """Set total steps at training start."""
        # Try to find JoraModel from the original model first
        jora_model = self._get_jora_model()

        # If not found, try to find it from trainer (in case model was wrapped)
        if jora_model is None and hasattr(kwargs.get('model', None), 'modules'):
            jora_model = self._find_jora_model_in_trainer(kwargs.get('model'))
        elif jora_model is None and hasattr(state, 'model'):
            jora_model = self._find_jora_model_in_trainer(state.model)

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

        self._initialized = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState",
                    control: "TrainerControl", **kwargs):
        """Trigger JORA updates at each optimizer step."""
        if not self._initialized:
            return

        jora_model = self._get_jora_model()
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


class JoraSchedulerCallback:
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
