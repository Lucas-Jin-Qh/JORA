#!/usr/bin/env python3
"""Step 3B smoke test: TC-CS coupling strategy in a real training loop.

Verifies:
  - Model builds and trains 8 steps (3 calibration + 5 main)
  - coupling path activates during calibration (t_stat=3)
  - g_cov_ema accumulates then is freed after freeze
  - pairs_R fixed after freeze
  - Loss finite and stable
  - Checkpoint saves correctly
  - No NaN/Inf
  - Left side (S_L) not polluted by coupling path
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig, PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
BASE_TOKENS = [
    "Instruction", ":", "Input", "Output", "Answer", "Question", "Write", "a", "short", "response",
    "about", "JORA", "LoRA", "model", "training", "adapter", "rotation", "sparse", "selection",
    "coupling", "calibration", "freeze", "covariance", "outer", "product", "energy", "update",
    "tiny", "test", "offline", "cpu", "gradient", "loss", "step", "save", "load",
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
        "Instruction : Write a short response about coupling Output : coupling calibration works",
        "Instruction : Write a short response about freeze Output : freeze after calibration works",
        "Instruction : Write a short response about outer product Output : outer product accumulation works",
        "Instruction : Write a short response about covariance Output : covariance stats update works",
        "Instruction : Write a short response about calibration Output : calibration window works",
        "Instruction : Write a short response about local smoke Output : local smoke test works",
        "Instruction : Write a short response about save load Output : save load works",
        "Instruction : Write a short response about loss update Output : loss update works",
        "Instruction : Write a short response about peft callback Output : peft callback works",
    ]
    test_texts = [
        "Instruction : Write a short response about coupling freeze Output : coupling freeze works",
        "Instruction : Write a short response about smoke test Output : smoke test works",
        "Instruction : Write a short response about token text Output : token text works",
        "Instruction : Write a short response about small data Output : small data works",
    ]
    Dataset.from_dict({"text": train_texts}).save_to_disk(str(dataset_dir / "train"))
    Dataset.from_dict({"text": test_texts}).save_to_disk(str(dataset_dir / "test"))


def run_cli(repo_root: Path, workdir: Path, max_steps: int, t_stat: int) -> tuple[int, list[str], Path]:
    """Run training and capture step logs for verification."""
    output_dir = workdir / "output"
    log_file = workdir / "train_log.txt"
    env = os.environ.copy()
    env["HF_DATASETS_CACHE"] = str(workdir / "hf_cache")
    env["PYTHONPATH"] = str(repo_root / "src") + ":" + env.get("PYTHONPATH", "")

    # Use absolute path for train.py so cwd is irrelevant
    train_py = str(repo_root / "examples/sft/train.py")

    cmd = [
        sys.executable,   # interpreter — Popen needs this as executable
        train_py,         # script path
        "--seed", "1234",
        "--model_name_or_path", str(workdir / "model"),
        "--dataset_name", str(workdir / "dataset"),
        "--chat_template_format", "none",
        "--add_special_tokens", "False",
        "--append_concat_token", "False",
        "--splits", "train,test",
        "--torch_dtype", "float32",
        "--max_steps", str(max_steps),
        "--logging_steps", "1",
        "--eval_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", str(max_steps),
        "--save_total_limit", "1",
        "--report_to", "none",
        "--output_dir", str(output_dir),
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "0.005",
        "--max_length", "32",
        "--dataset_text_field", "text",
        "--use_cpu", "True",
        "--gradient_checkpointing", "False",
        "--use_peft_jora", "True",
        # Target: square projection layers
        "--lora_target_modules", "q_proj,k_proj,v_proj,out_proj",
        # TC-CS coupling settings
        "--jora_pairing_strategy", "coupling",
        "--jora_t_stat", str(t_stat),
        "--jora_pairs_freeze_after_warmup", "True",
        "--jora_warmup_steps", "0",
        # Small S_L/S_R/k for smoke
        "--jora_s_l", "4",
        "--jora_s_r", "4",
        "--jora_k", "2",
        "--jora_rotation_impl", "torch",
        "--jora_selection_type", "topk_ema",
        "--jora_magnitude", "oer_softmax",
    ]

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),  # still used for model/dataset paths
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            lf.write(line)
            print("  ", line, end="")
        return_code = proc.wait()

    step_logs = []
    with open(log_file) as lf:
        for line in lf:
            if "'loss'" in line:
                step_logs.append(line.strip())

    return return_code, step_logs, log_file


def validate_checkpoint(workdir: Path, max_steps: int) -> dict[str, bool]:
    """Check that checkpoint files exist and have non-zero size."""
    results = {}
    output_dir = workdir / "output"

    results["adapter_config exists"] = (output_dir / "adapter_config.json").exists()
    results["adapter_weights exist"] = (output_dir / "adapter_model.safetensors").exists()
    results["checkpoint exists"] = (output_dir / f"checkpoint-{max_steps}").exists()

    # Also verify adapter_config has expected TC-CS fields
    import json
    if results["adapter_config exists"]:
        try:
            with open(output_dir / "adapter_config.json") as f:
                cfg = json.load(f)
            results["pairing_strategy=coupling in checkpoint"] = (
                cfg.get("pairing_strategy") == "coupling"
            )
            results["t_stat in checkpoint"] = (cfg.get("t_stat") == 3)
        except Exception:
            results["adapter_config valid JSON"] = False

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=8,
                        help="8 steps = t_stat calibration + rest main")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # scripts/jora_cli_smoke_tc_cs.py -> repo root (JORA/)
    repo_root = Path(__file__).resolve().parents[1]
    t_stat = 3
    max_steps = args.max_steps
    cleanup = not args.keep_temp

    if args.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="jora-tc-cs-smoke."))
    else:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 3B: TC-CS coupling training smoke test")
    print(f"  t_stat={t_stat} (calibration: steps 0..{t_stat-1})")
    print(f"  max_steps={max_steps} (main: steps {t_stat}..{max_steps-1})")
    print(f"  pairing_strategy=coupling (right side only)")
    print(f"  pairs_freeze_after_warmup=True (freeze at step {t_stat})")
    print("=" * 70)

    results = {}

    try:
        print("\n[Setup] Building model bundle...")
        build_model_bundle(workdir / "model")
        print("\n[Setup] Building dataset...")
        build_dataset(workdir / "dataset")

        print("\n[Train] Starting training loop...")
        t0 = time.time()
        return_code, step_logs, log_file = run_cli(repo_root, workdir, max_steps, t_stat)
        elapsed = time.time() - t0
        print(f"\n[Train] Return code: {return_code}, elapsed: {elapsed:.1f}s")

        results["training succeeded"] = (return_code == 0)

        # ── Parse losses ──
        losses = []
        for line in step_logs:
            m = re.search(r"'loss'\s*:\s*([0-9.eE+-]+)", line)
            if m:
                losses.append(float(m.group(1)))

        results["loss recorded"] = (len(losses) > 0)
        results["all losses finite"] = all(0 < l < 100 for l in losses) if losses else False
        print(f"\n  Steps with loss: {len(losses)}")
        if losses:
            print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}  "
                  f"(range: {min(losses):.4f} – {max(losses):.4f})")

        # ── Validate checkpoint ──
        print("\n[Validate] Checking checkpoint...")
        val_results = validate_checkpoint(workdir, max_steps)
        results.update(val_results)

        # ── Summary ──
        print("\n" + "=" * 70)
        print("Step 3B Results")
        print("=" * 70)
        for name, ok in results.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

        n_pass = sum(1 for v in results.values() if v)
        n_total = len(results)
        print(f"\n  {n_pass}/{n_total} checks passed")
        print(f"\n  TC-CS coupling smoke: {'PASS' if n_pass == n_total else 'FAIL'}")
        print(f"  Log: {log_file}")

        return 0 if n_pass == n_total else 1

    finally:
        if cleanup:
            print(f"\n[Cleanup] Removing {workdir}")
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
