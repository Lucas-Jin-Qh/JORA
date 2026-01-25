import os
from enum import Enum

import packaging.version
import torch
import transformers
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, JoraConfig


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}

    def preprocess_alpaca(samples):
        """预处理Alpaca数据集，将instruction+input+output组合成单个文本"""
        batch = []
        for instruction, input_text, output in zip(samples["instruction"], samples["input"], samples["output"]):
            if input_text.strip():  # 如果有输入
                text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            else:  # 如果没有输入
                text = f"Instruction: {instruction}\nOutput: {output}"
            batch.append(text)
        return {"text": batch}

    def preprocess_gsm8k(samples):
        """预处理GSM8K数据集，将question和answer组合成单个文本"""
        batch = []
        for question, answer in zip(samples["question"], samples["answer"]):
            # GSM8K格式：问题后面直接跟着答案
            text = f"Question: {question}\nAnswer: {answer}"
            batch.append(text)
        return {"text": batch}

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Parse dataset name to support config specification (e.g., "gsm8k:main")
            dataset_name = data_args.dataset_name
            config_name = None

            if ":" in dataset_name:
                dataset_name, config_name = dataset_name.split(":", 1)

            # Try first if dataset on a Hub repo
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split)
            else:
                # Try loading without config first, if it fails with ConfigError, try common configs
                try:
                    dataset = load_dataset(dataset_name, split=split)
                except ValueError as e:
                    if "Config name is missing" in str(e):
                        # Try common config names for datasets that require them
                        for config in ['main', 'socratic', 'default']:
                            try:
                                dataset = load_dataset(dataset_name, config, split=split)
                                print(f"Using config '{config}' for dataset '{dataset_name}'")
                                break
                            except Exception:
                                continue
                        else:
                            raise e  # Re-raise if no config worked
                    else:
                        raise e
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
    elif data_args.dataset_name == "yahma/alpaca-cleaned":
        # 特殊处理Alpaca数据集
        raw_datasets = raw_datasets.map(
            preprocess_alpaca,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
    elif "gsm8k" in data_args.dataset_name.lower():
        # 特殊处理GSM8K数据集
        raw_datasets = raw_datasets.map(
            preprocess_gsm8k,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    # Handle datasets without test split (e.g., Alpaca)
    if "test" in raw_datasets:
        valid_data = raw_datasets["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    else:
        # Use a portion of train data as validation if no test split exists
        train_valid_split = train_data.train_test_split(test_size=0.05, seed=42)
        train_data = train_valid_split["train"]
        valid_data = train_valid_split["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)} (created from train split)")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization and args.use_8bit_quantization:
        raise ValueError("You configured 4bit and 8bit quantization at the same time, please choose only one of them.")
    elif args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
    elif args.use_8bit_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        if torch.xpu.is_available():
            raise NotImplementedError("XPU hasn't supported unsloth yet")
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=training_args.max_seq_length or 2048,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        # Determine model dtype
        if args.torch_dtype == "auto":
            # Auto mode: use bfloat16 for flash attention, otherwise float32
            if args.use_flash_attn:
                dtype = torch.bfloat16
            else:
                dtype = quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        elif args.torch_dtype == "float16":
            dtype = torch.float16
        elif args.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.torch_dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported torch_dtype: {args.torch_dtype}")

        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": dtype,
        }
        if args.use_flash_attn:
            if torch.xpu.is_available():
                print("XPU hasn't supported flash_attn yet, use eager implementation instead.")
                model_kwargs["attn_implementation"] = "eager"
            else:
                model_kwargs["attn_implementation"] = "flash_attention_2"

        # Only add quantization_config if bnb_config is not None
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    peft_config = None
    chat_template = None
    if args.use_peft_jora and not args.use_peft_lora and not args.use_unsloth:
        peft_config = JoraConfig(
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            S_L=args.jora_s_l,
            S_R=args.jora_s_r,
            k=args.jora_k,
            rotation_param=args.jora_rotation_param,
            rotation_impl=args.jora_rotation_impl,
            selection=args.jora_selection_type,
            magnitude=args.jora_magnitude,
            update_interval=args.jora_update_interval,
            ema_update_interval=args.jora_ema_update_interval,
            selection_group_size=args.jora_selection_group_size,
            selection_group_by=args.jora_selection_group_by,
            inference_mode=False,
        )
    elif args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_dora=args.use_dora,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template

        # make embedding resizing configurable?
        # Transformers 4.46.0+ defaults uses mean_resizing by default, which fails with QLoRA + FSDP because the
        # embedding could be on meta device, therefore, we set mean_resizing=False in that case (i.e. the status quo
        # ante). See https://github.com/huggingface/accelerate/issues/1620.
        uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
        uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        # Check if the model is quantized
        is_quantized = (bnb_config is not None) or (getattr(model, "hf_quantizer", None) is not None)
        if is_quantized and uses_fsdp and uses_transformers_4_46:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        else:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=training_args.max_seq_length or 2048,
        )

    return model, peft_config, tokenizer
