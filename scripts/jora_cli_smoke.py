#!/usr/bin/env python3
"""Offline CLI smoke test for examples/sft/train.py with JORA."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig, PreTrainedTokenizerFast

from peft import PeftModel


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
BASE_TOKENS = [
    "Instruction", ":", "Input", "Output", "Answer", "Question", "Write", "a", "short", "response",
    "about", "JORA", "LoRA", "model", "training", "adapter", "rotation", "sparse", "selection", "energy",
    "tiny", "test", "offline", "cpu", "gradient", "update", "loss", "step", "save", "load",
    "hello", "world", "alpha", "beta", "gamma", "delta", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "small", "example", "data", "run",
    "works", "verify", "local", "smoke", "train", "eval", "peft", "callback", "token", "text",
]
VOCAB = SPECIAL_TOKENS + BASE_TOKENS


def build_model_bundle(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {token: idx for idx, token in enumerate(VOCAB)}

    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.decoder = WordPieceDecoder(prefix="")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.model_max_length = 64
    tokenizer.save_pretrained(model_dir)

    config = OPTConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        ffn_dim=64,
        max_position_embeddings=64,
        word_embed_proj_dim=32,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(model_dir)


def build_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_texts = [
        "Instruction : Write a short response about JORA Output : JORA adapter training works",
        "Instruction : Write a short response about LoRA Output : LoRA adapter update works",
        "Instruction : Write a short response about sparse selection Output : sparse selection works",
        "Instruction : Write a short response about rotation Output : rotation update works",
        "Instruction : Write a short response about energy Output : energy update works",
        "Instruction : Write a short response about tiny model Output : tiny model works",
        "Instruction : Write a short response about offline test Output : offline smoke test works",
        "Instruction : Write a short response about save load Output : save load works",
        "Instruction : Write a short response about callback step Output : callback step works",
        "Instruction : Write a short response about local train Output : local train works",
        "Instruction : Write a short response about cpu run Output : cpu run works",
        "Instruction : Write a short response about loss update Output : loss update works",
    ]
    test_texts = [
        "Instruction : Write a short response about local smoke Output : local smoke works",
        "Instruction : Write a short response about peft callback Output : peft callback works",
        "Instruction : Write a short response about token text Output : token text works",
        "Instruction : Write a short response about small data Output : small data works",
    ]
    Dataset.from_dict({"text": train_texts}).save_to_disk(str(dataset_dir / "train"))
    Dataset.from_dict({"text": test_texts}).save_to_disk(str(dataset_dir / "test"))


def run_cli(repo_root: Path, workdir: Path, max_steps: int) -> None:
    output_dir = workdir / "output"
    env = os.environ.copy()
    env["HF_DATASETS_CACHE"] = str(workdir / "hf_cache")
    cmd = [
        sys.executable,
        "examples/sft/train.py",
        "--seed",
        "1234",
        "--model_name_or_path",
        str(workdir / "model"),
        "--dataset_name",
        str(workdir / "dataset"),
        "--chat_template_format",
        "none",
        "--add_special_tokens",
        "False",
        "--append_concat_token",
        "False",
        "--splits",
        "train,test",
        "--torch_dtype",
        "float32",
        "--max_steps",
        str(max_steps),
        "--logging_steps",
        "1",
        "--eval_strategy",
        "no",
        "--save_strategy",
        "no",
        "--report_to",
        "none",
        "--output_dir",
        str(output_dir),
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "0.005",
        "--max_length",
        "32",
        "--dataset_text_field",
        "text",
        "--use_cpu",
        "True",
        "--gradient_checkpointing",
        "False",
        "--use_peft_jora",
        "True",
        "--lora_target_modules",
        "q_proj,k_proj,v_proj,out_proj",
        "--jora_s_l",
        "4",
        "--jora_s_r",
        "4",
        "--jora_k",
        "2",
        "--jora_rotation_impl",
        "torch",
        "--jora_selection_type",
        "topk_ema",
        "--jora_magnitude",
        "oer_softmax",
        "--jora_warmup_steps",
        "2",
    ]
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def validate_output(workdir: Path) -> None:
    model_dir = workdir / "model"
    output_dir = workdir / "output"
    adapter_config = output_dir / "adapter_config.json"
    adapter_weights = output_dir / "adapter_model.safetensors"
    if not adapter_config.exists():
        raise RuntimeError(f"Missing adapter config: {adapter_config}")
    if not adapter_weights.exists():
        raise RuntimeError(f"Missing adapter weights: {adapter_weights}")

    base_model = AutoModelForCausalLM.from_pretrained(model_dir)
    peft_model = PeftModel.from_pretrained(base_model, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    peft_model.eval()
    inputs = tokenizer("Instruction : Write a short response about JORA Output :", return_tensors="pt")
    with torch.no_grad():
        outputs = peft_model(**inputs)
    if outputs.logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected logits shape: {tuple(outputs.logits.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=None, help="Optional working directory to keep artifacts.")
    parser.add_argument("--max-steps", type=int, default=4, help="Trainer max_steps for the smoke run.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary workdir after success.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cleanup = False
    if args.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="jora-cli-smoke."))
        cleanup = not args.keep_temp
    else:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        build_model_bundle(workdir / "model")
        build_dataset(workdir / "dataset")
        run_cli(repo_root, workdir, args.max_steps)
        validate_output(workdir)
        print(f"Smoke test passed. Artifacts: {workdir}")
        return 0
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
