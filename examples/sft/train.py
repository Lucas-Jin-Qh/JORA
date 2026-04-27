import os
import sys
import inspect
import warnings
# Handle offline HF mode (when HuggingFace Hub is not accessible)
if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    # Try to detect if HF is accessible; if not, fall back to offline
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=2)
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("[JORA] HuggingFace Hub not accessible, using offline mode", flush=True)

from dataclasses import dataclass, field
from typing import Optional

# 过滤无害警告
warnings.filterwarnings("ignore")

from transformers import HfArgumentParser, set_seed
from trl import SFTConfig, SFTTrainer
from utils import create_and_prepare_model, create_datasets


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    jora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of target modules to apply JORA layers to. Defaults to q_proj,k_proj,v_proj,o_proj for the selective_diag paper path."
        },
    )
    use_peft_jora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT JORA for training."},
    )
    jora_s_l: Optional[int] = field(
        default=32,
        metadata={"help": "Left rotation matrix dimension for JORA."},
    )
    jora_s_r: Optional[int] = field(
        default=32,
        metadata={"help": "Right rotation matrix dimension for JORA."},
    )
    jora_k: Optional[int] = field(
        default=8,
        metadata={"help": "Selection parameter k for JORA sparse selection."},
    )
    jora_rotation_param: Optional[str] = field(
        default="cayley",
        metadata={"help": "Rotation parameterization for JORA ('cayley' or 'angle')."},
    )
    jora_rotation_impl: Optional[str] = field(
        default="auto",
        metadata={"help": "Rotation implementation for JORA ('auto', 'torch', 'triton')."},
    )
    jora_selection_type: Optional[str] = field(
        default="topk_ema",
        metadata={"help": "Parameter selection type for JORA ('topk_ema', 'random', 'none')."},
    )
    jora_magnitude: Optional[str] = field(
        default="none",
        metadata={"help": "Magnitude scaling type for JORA ('ecd_tanh', 'oer_softmax', 'none')."},
    )
    jora_update_interval: Optional[int] = field(
        default=1,
        metadata={"help": "Update interval for JORA parameter selection."},
    )
    jora_ema_update_interval: Optional[int] = field(
        default=1,
        metadata={"help": "EMA update interval for JORA."},
    )
    jora_selection_group_size: Optional[int] = field(
        default=1,
        metadata={"help": "Group size for JORA selection sharing (1 = per-module, >1 = grouped)."},
    )
    jora_selection_group_by: Optional[str] = field(
        default="dimension",
        metadata={"help": "Grouping strategy for JORA selection ('dimension', 'type', 'none')."},
    )
    # P0 JORA core parameters
    jora_core: Optional[str] = field(
        default="selective_diag",
        metadata={"help": "JORA core type ('diag', 'block', 'lowrank', 'selective_diag')."},
    )
    jora_block_size: Optional[int] = field(
        default=4,
        metadata={"help": "Block size for JORA 'block' core."},
    )
    jora_lowrank_r: Optional[int] = field(
        default=8,
        metadata={"help": "Rank for JORA 'lowrank' core."},
    )
    jora_lowrank_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Alpha for JORA 'lowrank' core."},
    )
    jora_zero_init_core: Optional[bool] = field(
        default=False,
        metadata={"help": "Zero-initialize JORA core parameters."},
    )
    # Paper-path calibration parameters
    jora_t_stat: Optional[int] = field(
        default=None,
        metadata={"help": "Number of calibration steps for JORA paper path (EMA collection before support freeze). None = use JoraConfig default."},
    )
    jora_pairs_freeze_after_warmup: Optional[bool] = field(
        default=None,
        metadata={"help": "Freeze JORA pairs after warmup completes (paper-path one-shot allocation). None = use JoraConfig default."},
    )
    # P0 JORA OER parameters
    jora_oer_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "Temperature for OER softmax magnitude."},
    )
    # P0 JORA schedule parameters
    jora_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Warmup steps for JORA pair selection."},
    )
    jora_warmup_ratio: Optional[float] = field(
        default=0.0,
        metadata={"help": "Warmup ratio for JORA pair selection."},
    )
    jora_single_sided: Optional[str] = field(
        default="none",
        metadata={"help": "Single-sided JORA ('none', 'left', 'right')."},
    )
    jora_pairing_strategy: Optional[str] = field(
        default="consecutive",
        metadata={"help": "Pairing strategy for JORA ('consecutive', 'high_low')."},
    )
    jora_ema_beta: Optional[float] = field(
        default=0.98,
        metadata={"help": "EMA beta for JORA activation tracking."},
    )
    jora_ema_grad_interval: Optional[int] = field(
        default=1,
        metadata={"help": "EMA gradient update interval for JORA."},
    )
    # P0 JORA learning rate parameters
    jora_lr_theta: Optional[float] = field(
        default=0.05,
        metadata={"help": "Learning rate for JORA theta (rotation) parameters."},
    )
    jora_lr_core: Optional[float] = field(
        default=0.01,
        metadata={"help": "Learning rate for JORA core parameters."},
    )
    # P0 JORA initialization parameters
    jora_theta_init_std: Optional[float] = field(
        default=None,
        metadata={"help": "Standard deviation for theta (rotation) parameter initialization. None = use JoraConfig default."},
    )
    jora_core_init_std: Optional[float] = field(
        default=None,
        metadata={"help": "Standard deviation for core parameter initialization. None = use JoraConfig default."},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_peft_oft: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT OFT for training."},
    )
    use_dora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables DoRA (Weight-Decomposed Low-Rank Adaptation) for LoRA training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    # OFT specific args
    oft_r: Optional[int] = field(default=0, metadata={"help": "OFT rank, number of OFT blocks per injected layer."})
    oft_block_size: Optional[int] = field(default=32, metadata={"help": "OFT block size across different layers."})
    oft_module_dropout: Optional[float] = field(default=0.0, metadata={"help": "OFT multiplicative dropout."})
    oft_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply OFT layers to"},
    )
    oft_fan_in_fan_out: Optional[bool] = field(default=False, metadata={"help": "Set to True if layer stores weight like (fan_in, fan_out)."})
    oft_bias: Optional[str] = field(default="none", metadata={"help": "Bias type for OFT: none|all|oft_only"})
    oft_init_weights: Optional[bool] = field(default=True, metadata={"help": "Whether to initialize OFT weights."})
    oft_layers_to_transform: Optional[str] = field(default=None, metadata={"help": "Comma separated layer indices to transform."})
    oft_layers_pattern: Optional[str] = field(default=None, metadata={"help": "Layer pattern name if using layers_to_transform."})
    oft_modules_to_save: Optional[str] = field(default=None, metadata={"help": "Comma separated modules to save apart from OFT layers."})
    oft_coft: Optional[bool] = field(default=False, metadata={"help": "Whether to use constrained OFT (COFT)."})
    oft_eps: Optional[float] = field(default=6e-5, metadata={"help": "COFT control strength."})
    oft_block_share: Optional[bool] = field(default=False, metadata={"help": "Whether to share OFT parameters between blocks."})
    oft_use_cayley_neumann: Optional[bool] = field(default=True, metadata={"help": "Use Cayley-Neumann formulation."})
    oft_num_cayley_neumann_terms: Optional[int] = field(default=5, metadata={"help": "Number of Cayley-Neumann terms."})
    # BOFT specific args
    use_peft_boft: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT BOFT (Butterfly OFT) for training."},
    )
    boft_block_size: Optional[int] = field(default=4, metadata={"help": "BOFT block size across different layers."})
    boft_block_num: Optional[int] = field(default=0, metadata={"help": "Number of BOFT blocks per injected layer."})
    boft_n_butterfly_factor: Optional[int] = field(default=1, metadata={"help": "Number of butterfly factors."})
    boft_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply BOFT layers to"},
    )
    boft_fan_in_fan_out: Optional[bool] = field(default=False, metadata={"help": "Set to True if layer stores weight like (fan_in, fan_out)."})
    boft_bias: Optional[str] = field(default="none", metadata={"help": "Bias type for BOFT: none|all|boft_only"})
    boft_init_weights: Optional[bool] = field(default=True, metadata={"help": "Whether to initialize BOFT weights."})
    boft_dropout: Optional[float] = field(default=0.0, metadata={"help": "BOFT multiplicative dropout."})
    boft_layers_to_transform: Optional[str] = field(default=None, metadata={"help": "Comma separated layer indices to transform."})
    boft_layers_pattern: Optional[str] = field(default=None, metadata={"help": "Layer pattern name if using layers_to_transform."})
    boft_modules_to_save: Optional[str] = field(default=None, metadata={"help": "Comma separated modules to save apart from BOFT layers."})
    # IA3 specific args
    use_peft_ia3: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT IA3 for training."},
    )
    ia3_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply IA3 layers to"},
    )
    ia3_feedforward_modules: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of modules to be treated as feedforward modules"},
    )
    ia3_fan_in_fan_out: Optional[bool] = field(default=False, metadata={"help": "Set to True if layer stores weight like (fan_in, fan_out)."})
    ia3_modules_to_save: Optional[str] = field(default=None, metadata={"help": "Comma separated modules to save apart from IA3 layers."})
    ia3_init_weights: Optional[bool] = field(default=True, metadata={"help": "Whether to initialize the vectors in the IA3 layers."})
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Model dtype: auto, float16, bfloat16, float32"},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def _configure_jora_optimizer_groups(trainer, model_args, training_args):
    """Rebuild optimizer with JORA-aware param groups.

    Strategy: hybrid reconstruction
      - JORA params (theta/core/magnitude): use JoraModel.get_optimizer_param_groups()
        which provides correct 'name' fields for callback detection
      - All other trainable params (e.g. modules_to_save): collect into 'jora_other'

    Success criteria:
      - Callback on_train_begin does NOT warn about missing 'jora_theta'
      - ALL requires_grad=True params are in some group (hard assertion, not just print)
    """
    if not model_args.use_peft_jora:
        return

    # Determine LR overrides from CLI args
    lr_theta = model_args.jora_lr_theta if model_args.jora_lr_theta is not None else training_args.learning_rate
    lr_core = model_args.jora_lr_core if model_args.jora_lr_core is not None else training_args.learning_rate

    if trainer.optimizer is None:
        trainer.create_optimizer()

    optimizer = trainer.optimizer
    if optimizer is None:
        return

    # Collect optimizer constructor kwargs
    default_weight_decay = float(optimizer.defaults.get("weight_decay", 0.0))
    optimizer_defaults = dict(optimizer.defaults)
    optimizer_signature = inspect.signature(optimizer.__class__.__init__)
    valid_optimizer_kwargs = {
        name
        for name, parameter in optimizer_signature.parameters.items()
        if name not in {"self", "params"}
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    optimizer_kwargs = {key: value for key, value in optimizer_defaults.items() if key in valid_optimizer_kwargs}

    # Collect which params should decay (from trainer's decay logic)
    decay_parameter_names = set()
    get_decay_parameter_names = getattr(trainer, "get_decay_parameter_names", None)
    if callable(get_decay_parameter_names):
        decay_parameter_names = set(get_decay_parameter_names(trainer.model))

    # --- Step 1: Get JORA param groups from helper ---
    from peft.tuners.jora.model import JoraModel

    jora_model = None
    def find_jora_model(m):
        if isinstance(m, JoraModel):
            return m
        if hasattr(m, "base_model") and isinstance(m.base_model, JoraModel):
            return m.base_model
        for child in m.children():
            result = find_jora_model(child)
            if result is not None:
                return result
        return None

    jora_model = find_jora_model(trainer.model)

    if jora_model is None or not hasattr(jora_model, "get_optimizer_param_groups"):
        trainer.accelerator.print(
            "[JORA] Warning: JoraModel not found. "
            "Skipping JORA-specific param groups. "
            "lr_theta and lr_core config values will be ignored."
        )
        return

    base_lr = training_args.learning_rate
    jora_groups = jora_model.get_optimizer_param_groups(base_lr=base_lr)

    # Apply CLI LR overrides to JORA groups
    for g in jora_groups:
        if g.get("name") == "jora_theta":
            g["lr"] = lr_theta
        elif g.get("name") == "jora_core":
            g["lr"] = lr_core
        # magnitude uses default magnitude_lr_scale (no CLI override)

    # Build a set of JORA param pointers for fast membership check
    jora_param_ptrs: set[int] = set()
    for g in jora_groups:
        for p in g["params"]:
            jora_param_ptrs.add(id(p))

    # --- Step 2: Collect non-JORA trainable params into 'other' group ---
    all_requires_grad_params: list[tuple[str, torch.nn.Parameter]] = []
    other_params_decayed: list[torch.nn.Parameter] = []
    other_params_no_decay: list[torch.nn.Parameter] = []

    for name, param in trainer.model.named_parameters():
        if not param.requires_grad:
            continue
        all_requires_grad_params.append((name, param))
        if id(param) not in jora_param_ptrs:
            use_decay = name in decay_parameter_names
            if use_decay:
                other_params_decayed.append(param)
            else:
                other_params_no_decay.append(param)

    # Build 'jora_other' group(s)
    new_param_groups: list[dict] = []
    for g in jora_groups:
        # Apply weight_decay split based on trainer's decay policy
        name = g.get("name", "unknown")
        params_with_decay: list[torch.nn.Parameter] = []
        params_no_decay: list[torch.nn.Parameter] = []

        for p in g["params"]:
            # Find the param's full name in the model for decay check
            param_name_in_model = None
            for n, _ in trainer.model.named_parameters():
                if id(p) == id(_find_param(trainer.model, n)):
                    param_name_in_model = n
                    break
            if param_name_in_model is not None and param_name_in_model in decay_parameter_names:
                params_with_decay.append(p)
            else:
                params_no_decay.append(p)

        if params_with_decay:
            new_param_groups.append({
                "params": params_with_decay,
                "lr": g["lr"],
                "weight_decay": default_weight_decay,
                "name": name,
            })
        if params_no_decay:
            new_param_groups.append({
                "params": params_no_decay,
                "lr": g["lr"],
                "weight_decay": 0.0,
                "name": name,
            })

    if other_params_decayed:
        new_param_groups.append({
            "params": other_params_decayed,
            "lr": training_args.learning_rate,
            "weight_decay": default_weight_decay,
            "name": "jora_other",
        })
    if other_params_no_decay:
        new_param_groups.append({
            "params": other_params_no_decay,
            "lr": training_args.learning_rate,
            "weight_decay": 0.0,
            "name": "jora_other",
        })

    # --- Step 3: Hard assertion — coverage must be 100% ---
    grouped_param_ptrs: set[int] = set()
    for g in new_param_groups:
        for p in g["params"]:
            grouped_param_ptrs.add(id(p))

    missing = []
    for name, param in all_requires_grad_params:
        if id(param) not in grouped_param_ptrs:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"[JORA] Optimizer coverage gap! {len(missing)} trainable params are not in any group: {missing[:5]}"
        )

    # Also assert: no param should appear in multiple groups
    duplicates = []
    for g in new_param_groups:
        for p in g["params"]:
            ptr = id(p)
            if ptr in grouped_param_ptrs and ptr not in jora_param_ptrs:
                # first occurrence
                pass
    # Simpler: count total
    total_in_groups = sum(len(g["params"]) for g in new_param_groups)
    assert total_in_groups == len(all_requires_grad_params), (
        f"[JORA] Param count mismatch: groups have {total_in_groups} params, "
        f"but model has {len(all_requires_grad_params)} trainable params. "
        "Possible duplicate param assignment."
    )

    # --- Step 4: Rebuild optimizer ---
    trainer.optimizer = optimizer.__class__(new_param_groups, **optimizer_kwargs)

    # --- Step 5: Print summary ---
    trainer.accelerator.print("=" * 60)
    trainer.accelerator.print("JORA Optimizer Param Groups (hybrid reconstruction)")
    trainer.accelerator.print("=" * 60)
    for g in new_param_groups:
        param_count = sum(p.numel() for p in g["params"])
        trainer.accelerator.print(
            f"  {g.get('name', 'unnamed')}: {param_count:,} params, "
            f"lr={g['lr']:.2e}, weight_decay={g.get('weight_decay', 0.0):.1e}"
        )
    total_trainable = sum(sum(p.numel() for p in g["params"]) for g in new_param_groups)
    trainer.accelerator.print(f"  Total trainable: {total_trainable:,}")
    trainer.accelerator.print(
        f"  Coverage: {len(all_requires_grad_params)}/{len(all_requires_grad_params)} = 100% [ASSERTED]"
    )
    trainer.accelerator.print("=" * 60)
    trainer.accelerator.print(
        f"[JORA] Applied JORA-specific LR: theta={lr_theta}, core={lr_core}"
    )


def _find_param(model, name: str):
    """Helper: find parameter by name in model."""
    for n, p in model.named_parameters():
        if n == name:
            return p
    return None


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    training_args.dataset_kwargs = {
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    }

    # datasets
    train_dataset, eval_dataset = create_datasets(
        tokenizer,
        data_args,
        training_args,
        apply_chat_template=model_args.chat_template_format != "none",
    )

    # JORA requires special DDP settings for sparse parameter selection
    if model_args.use_peft_jora:
        # Set DDP parameters for JORA's sparse selection mechanism
        training_args.ddp_find_unused_parameters = True
        if hasattr(training_args, 'ddp_timeout'):
            training_args.ddp_timeout = 1800  # 30 minutes timeout for sparse updates

    # trainer
    callbacks = []
    if model_args.use_peft_jora:
        # Import JORA callback for reliable updates
        from peft.tuners.jora.callbacks import JoraTrainerCallback, JoraMetricsCallback
        callbacks.append(JoraTrainerCallback(model, verbose=False))
        callbacks.append(JoraMetricsCallback(model, output_dir=training_args.output_dir, log_interval=training_args.logging_steps))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    _configure_jora_optimizer_groups(trainer, model_args, training_args)

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # Strip metadata keys (starting with '_') before parsing to avoid HfArgumentParser errors.
        import json
        with open(os.path.abspath(sys.argv[1])) as f:
            raw_data = json.load(f)
        clean_data = {k: v for k, v in raw_data.items() if not k.startswith("_")}
        model_args, data_args, training_args = parser.parse_dict(clean_data, allow_extra_keys=False)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
