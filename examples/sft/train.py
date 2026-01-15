import os
import sys
import warnings
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
        default="ecd_tanh",
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
        from peft.tuners.jora.callbacks import JoraTrainerCallback
        callbacks.append(JoraTrainerCallback(model, verbose=False))

    # Add callbacks to training args instead of passing directly to SFTTrainer
    if callbacks:
        training_args.callbacks = callbacks

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

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
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
