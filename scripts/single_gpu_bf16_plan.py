#!/usr/bin/env python3
"""Single-GPU bf16 experiment launcher for the claim-bearing JORA plan.

This script does not modify training codepaths. It builds and optionally runs
explicit `examples/sft/train.py` commands for the agreed single-GPU ladder:

- M0 sanity on OPT-350M
- M1 JORA LR sweep on OPT-350M
- M2a bf16 feasibility probes on Mistral-7B
- M2b single-seed anchor runs on Mistral-7B

The intent is to keep the broad legacy stage scripts untouched while providing
one narrow, reproducible path for the paper-claim experiments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "examples" / "sft" / "train.py"
DEFAULT_WORKDIR = REPO_ROOT / "formal_runs" / "single_gpu_bf16"
DEFAULT_SHARED_CACHE_ROOT = REPO_ROOT / "formal_runs" / "shared_hf_cache"
DEFAULT_SHARED_HF_HOME = DEFAULT_SHARED_CACHE_ROOT / "hf_home"
DEFAULT_SHARED_HF_DATASETS_CACHE = DEFAULT_SHARED_CACHE_ROOT / "hf_datasets"
LEGACY_FORMAL_HF_HOME = REPO_ROOT / "formal_runs" / "three_gpu_bf16" / "hf_home"
LEGACY_FORMAL_HF_DATASETS_CACHE = REPO_ROOT / "formal_runs" / "three_gpu_bf16" / "hf_datasets"
MISTRAL_7B_MODEL_ID = "mistralai/Mistral-7B-v0.1"
LOCAL_MISTRAL_7B_PATH = Path("/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1")
RETRYABLE_LAUNCH_FILES = {"run_spec.json", "run_command.sh"}
COMMON_DATASET = "yahma/alpaca-cleaned"
COMMON_TARGET_MODULES = "q_proj,o_proj"
DEFAULT_JORA_LR_THETA = 5e-3
DEFAULT_JORA_LR_CORE = 1e-3
SAME_BUDGET_TOLERANCE = 0.15


@dataclass(frozen=True)
class TrainSpec:
    name: str
    description: str
    model_name_or_path: str
    output_subdir: str
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    logging_steps: int
    method: str
    max_steps: int | None = None
    num_train_epochs: float | None = None
    save_strategy: str = "no"
    save_steps: int | None = None
    save_total_limit: int | None = None
    seed: int = 42
    dataset_name: str = COMMON_DATASET
    splits: str = "train"
    lora_r: int | None = None
    lora_alpha: int | None = None
    jora_core: str = "selective_diag"
    jora_block_size: int | None = None
    jora_lowrank_r: int | None = None
    jora_lowrank_alpha: float | None = None
    jora_zero_init_core: bool | None = None
    jora_selection_type: str = "topk_ema"
    jora_s_l: int = 32
    jora_s_r: int = 32
    jora_k: int = 32
    jora_t_stat: int = 200
    jora_pairs_freeze_after_warmup: bool = True
    jora_magnitude: str = "none"
    jora_lr_theta: float = DEFAULT_JORA_LR_THETA
    jora_lr_core: float = DEFAULT_JORA_LR_CORE


def _format_float_token(value: float) -> str:
    return f"{value:g}".replace("+", "")


def default_mistral_model_name_or_path() -> str:
    if LOCAL_MISTRAL_7B_PATH.exists():
        return str(LOCAL_MISTRAL_7B_PATH)
    return MISTRAL_7B_MODEL_ID


def _base_named_specs() -> dict[str, TrainSpec]:
    mistral_model = default_mistral_model_name_or_path()
    return {
        "m0_jora": TrainSpec(
            name="m0_jora",
            description="OPT-350M sanity run for paper-path JORA on q_proj,o_proj.",
            model_name_or_path="facebook/opt-350m",
            output_subdir="m0/jora_base_seed42",
            max_steps=500,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            max_length=512,
            logging_steps=25,
            method="jora",
        ),
        "m0_lora_r1": TrainSpec(
            name="m0_lora_r1",
            description="OPT-350M sanity run for LoRA-r1 on q_proj,o_proj.",
            model_name_or_path="facebook/opt-350m",
            output_subdir="m0/lora_r1_seed42",
            max_steps=500,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            max_length=512,
            logging_steps=25,
            method="lora",
            lora_r=1,
            lora_alpha=2,
        ),
        "m2a_jora": TrainSpec(
            name="m2a_jora",
            description="Mistral-7B pure-bf16 feasibility probe for JORA-base.",
            model_name_or_path=mistral_model,
            output_subdir="m2a/jora_base_seed42",
            max_steps=100,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=512,
            logging_steps=10,
            method="jora",
            save_strategy="steps",
            save_steps=50,
            save_total_limit=1,
        ),
        "m2a_lora_r1": TrainSpec(
            name="m2a_lora_r1",
            description="Mistral-7B pure-bf16 feasibility probe for LoRA-r1.",
            model_name_or_path=mistral_model,
            output_subdir="m2a/lora_r1_seed42",
            max_steps=100,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=512,
            logging_steps=10,
            method="lora",
            lora_r=1,
            lora_alpha=2,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=1,
        ),
        "m2b_jora": TrainSpec(
            name="m2b_jora",
            description="Mistral-7B single-seed anchor run for JORA-base.",
            model_name_or_path=mistral_model,
            output_subdir="m2b/jora_base_seed42",
            max_steps=2000,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=512,
            logging_steps=50,
            method="jora",
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
        ),
        "m2b_lora_r1": TrainSpec(
            name="m2b_lora_r1",
            description="Mistral-7B single-seed anchor run for LoRA-r1.",
            model_name_or_path=mistral_model,
            output_subdir="m2b/lora_r1_seed42",
            max_steps=2000,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=512,
            logging_steps=50,
            method="lora",
            lora_r=1,
            lora_alpha=2,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
        ),
        "m2b_lora_r2": TrainSpec(
            name="m2b_lora_r2",
            description="Mistral-7B single-seed anchor run for LoRA-r2.",
            model_name_or_path=mistral_model,
            output_subdir="m2b/lora_r2_seed42",
            max_steps=2000,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=512,
            logging_steps=50,
            method="lora",
            lora_r=2,
            lora_alpha=4,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
        ),
        "m2b_fixed_jora": TrainSpec(
            name="m2b_fixed_jora",
            description="Mistral-7B single-seed fixed-slot baseline via random frozen support.",
            model_name_or_path=mistral_model,
            output_subdir="m2b/jora_fixed_random_seed42",
            max_steps=2000,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=512,
            logging_steps=50,
            method="jora",
            jora_selection_type="random",
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
        ),
    }


def _m1_sweep_specs() -> list[TrainSpec]:
    specs = []
    lr_theta_values = [1e-3, 5e-3, 1e-2, 5e-2]
    lr_core_values = [5e-4, 1e-3, 5e-3]
    for lr_theta, lr_core in itertools.product(lr_theta_values, lr_core_values):
        token_theta = _format_float_token(lr_theta)
        token_core = _format_float_token(lr_core)
        name = f"m1_theta_{token_theta}_core_{token_core}"
        specs.append(
            TrainSpec(
                name=name,
                description="OPT-350M LR screening run for paper-path JORA on q_proj,o_proj.",
                model_name_or_path="facebook/opt-350m",
                output_subdir=f"m1/{name}_seed42",
                max_steps=500,
                learning_rate=1e-4,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                max_length=512,
                logging_steps=25,
                method="jora",
                jora_lr_theta=lr_theta,
                jora_lr_core=lr_core,
            )
        )
    return specs


def experiment_names() -> list[str]:
    return sorted(list(_base_named_specs()) + ["m1_sweep"])


def _apply_overrides(spec: TrainSpec, args: argparse.Namespace) -> TrainSpec:
    updates = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.model_name_or_path is not None:
        updates["model_name_or_path"] = args.model_name_or_path
    if args.dataset_name is not None:
        updates["dataset_name"] = args.dataset_name
    if args.max_steps is not None:
        updates["max_steps"] = args.max_steps
        updates["num_train_epochs"] = None
    if args.num_train_epochs is not None:
        updates["num_train_epochs"] = args.num_train_epochs
        updates["max_steps"] = None
    if args.learning_rate is not None:
        updates["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        updates["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        updates["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_length is not None:
        updates["max_length"] = args.max_length
    if spec.method == "jora":
        if args.jora_lr_theta is not None:
            updates["jora_lr_theta"] = args.jora_lr_theta
        if args.jora_lr_core is not None:
            updates["jora_lr_core"] = args.jora_lr_core
    return replace(spec, **updates)


def specs_for_name(name: str, args: argparse.Namespace) -> list[TrainSpec]:
    if name == "m1_sweep":
        return [_apply_overrides(spec, args) for spec in _m1_sweep_specs()]
    named = _base_named_specs()
    return [_apply_overrides(named[name], args)]


def build_train_command(spec: TrainSpec, output_dir: Path, python_bin: str) -> list[str]:
    if (spec.max_steps is None) == (spec.num_train_epochs is None):
        raise ValueError(
            f"TrainSpec {spec.name} must set exactly one of max_steps or num_train_epochs, "
            f"got max_steps={spec.max_steps} num_train_epochs={spec.num_train_epochs}."
        )

    command = [
        python_bin,
        str(TRAIN_SCRIPT),
        "--seed",
        str(spec.seed),
        "--model_name_or_path",
        spec.model_name_or_path,
        "--dataset_name",
        spec.dataset_name,
        "--chat_template_format",
        "none",
        "--add_special_tokens",
        "False",
        "--append_concat_token",
        "False",
        "--splits",
        spec.splits,
        "--torch_dtype",
        "bfloat16",
        "--bf16",
        "True",
        "--logging_steps",
        str(spec.logging_steps),
        "--eval_strategy",
        "no",
        "--save_strategy",
        spec.save_strategy,
        "--report_to",
        "none",
        "--output_dir",
        str(output_dir),
        "--per_device_train_batch_size",
        str(spec.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(spec.gradient_accumulation_steps),
        "--learning_rate",
        f"{spec.learning_rate:g}",
        "--max_length",
        str(spec.max_length),
        "--dataset_text_field",
        "text",
        "--use_cpu",
        "False",
        "--gradient_checkpointing",
        "True",
        "--use_reentrant",
        "False",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.03",
    ]

    if spec.max_steps is not None:
        command.extend(["--max_steps", str(spec.max_steps)])
    else:
        command.extend(["--num_train_epochs", f"{spec.num_train_epochs:g}"])

    if spec.save_strategy == "steps":
        if spec.save_steps is None or spec.save_total_limit is None:
            raise ValueError(f"steps save strategy requires save_steps/save_total_limit: {spec.name}")
        command.extend(
            [
                "--save_steps",
                str(spec.save_steps),
                "--save_total_limit",
                str(spec.save_total_limit),
            ]
        )
    elif spec.save_total_limit is not None:
        command.extend(["--save_total_limit", str(spec.save_total_limit)])

    if spec.method == "jora":
        command.extend(
            [
                "--use_peft_jora",
                "True",
                "--jora_core",
                spec.jora_core,
                "--jora_target_modules",
                COMMON_TARGET_MODULES,
                "--jora_magnitude",
                spec.jora_magnitude,
                "--jora_t_stat",
                str(spec.jora_t_stat),
                "--jora_pairs_freeze_after_warmup",
                "True" if spec.jora_pairs_freeze_after_warmup else "False",
                "--jora_selection_type",
                spec.jora_selection_type,
                "--jora_s_l",
                str(spec.jora_s_l),
                "--jora_s_r",
                str(spec.jora_s_r),
                "--jora_k",
                str(spec.jora_k),
                "--jora_lr_theta",
                f"{spec.jora_lr_theta:g}",
                "--jora_lr_core",
                f"{spec.jora_lr_core:g}",
            ]
        )
        if spec.jora_block_size is not None:
            command.extend(["--jora_block_size", str(spec.jora_block_size)])
        if spec.jora_lowrank_r is not None:
            command.extend(["--jora_lowrank_r", str(spec.jora_lowrank_r)])
        if spec.jora_lowrank_alpha is not None:
            command.extend(["--jora_lowrank_alpha", f"{spec.jora_lowrank_alpha:g}"])
        if spec.jora_zero_init_core is not None:
            command.extend(["--jora_zero_init_core", "True" if spec.jora_zero_init_core else "False"])
    elif spec.method == "lora":
        if spec.lora_r is None:
            raise ValueError(f"LoRA spec missing rank: {spec.name}")
        lora_alpha = spec.lora_alpha if spec.lora_alpha is not None else 2 * spec.lora_r
        command.extend(
            [
                "--use_peft_lora",
                "True",
                "--lora_target_modules",
                COMMON_TARGET_MODULES,
                "--lora_r",
                str(spec.lora_r),
                "--lora_alpha",
                str(lora_alpha),
                "--lora_dropout",
                "0.0",
            ]
        )
    else:
        raise ValueError(f"Unsupported method: {spec.method}")

    return command


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def shared_hf_home_path() -> Path:
    # Prefer the active formal rollout cache if it already exists, so resumed
    # and future formal runs reuse the in-flight Mistral download.
    if LEGACY_FORMAL_HF_HOME.exists():
        return LEGACY_FORMAL_HF_HOME
    return DEFAULT_SHARED_HF_HOME


def shared_hf_datasets_cache_path() -> Path:
    if LEGACY_FORMAL_HF_DATASETS_CACHE.exists():
        return LEGACY_FORMAL_HF_DATASETS_CACHE
    return DEFAULT_SHARED_HF_DATASETS_CACHE


def is_retryable_launch_dir(output_dir: Path) -> bool:
    if not output_dir.exists() or not output_dir.is_dir():
        return False
    files = {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()}
    return files.issubset(RETRYABLE_LAUNCH_FILES)


def prepare_output_dir_for_launch(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if is_retryable_launch_dir(output_dir):
        shutil.rmtree(output_dir)
        return
    raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")


def build_env(workdir: Path, hf_endpoint: str | None) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_HOME", str(shared_hf_home_path()))
    env.setdefault("HF_DATASETS_CACHE", str(shared_hf_datasets_cache_path()))
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    return env


def write_manifest(output_dir: Path, spec: TrainSpec, command: list[str], env: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "spec": asdict(spec),
        "claim_scope": {
            "target_modules": COMMON_TARGET_MODULES,
            "same_budget_tolerance": SAME_BUDGET_TOLERANCE,
            "dtype": "bfloat16",
            "uses_quantization": False,
        },
        "command": command,
        "command_text": format_command(command),
        "environment": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": env.get("PYTHONPATH"),
            "HF_HOME": env.get("HF_HOME"),
            "HF_DATASETS_CACHE": env.get("HF_DATASETS_CACHE"),
            "HF_HUB_DISABLE_XET": env.get("HF_HUB_DISABLE_XET"),
            "HF_ENDPOINT": env.get("HF_ENDPOINT"),
        },
    }
    (output_dir / "run_spec.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    export_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])}",
        f"export PYTHONPATH={shlex.quote(env['PYTHONPATH'])}",
        f"export HF_HOME={shlex.quote(env['HF_HOME'])}",
        f"export HF_DATASETS_CACHE={shlex.quote(env['HF_DATASETS_CACHE'])}",
        f"export HF_HUB_DISABLE_XET={shlex.quote(env['HF_HUB_DISABLE_XET'])}",
        f"export TOKENIZERS_PARALLELISM={shlex.quote(env['TOKENIZERS_PARALLELISM'])}",
    ]
    if env.get("HF_ENDPOINT"):
        export_lines.append(f"export HF_ENDPOINT={shlex.quote(env['HF_ENDPOINT'])}")
    export_lines.append(format_command(command))
    (output_dir / "run_command.sh").write_text("\n".join(export_lines) + "\n", encoding="utf-8")


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment", choices=experiment_names())
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--hf-endpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--dataset-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--jora-lr-theta", type=float)
    parser.add_argument("--jora-lr-core", type=float)


def list_experiments() -> None:
    specs = _base_named_specs()
    print("Single-GPU bf16 experiments:")
    for name in sorted(specs):
        print(f"  {name}: {specs[name].description}")
    print("  m1_sweep: OPT-350M JORA LR sweep over 12 theta/core combinations.")


def show_experiments(args: argparse.Namespace) -> None:
    env = build_env(args.workdir, args.hf_endpoint)
    specs = specs_for_name(args.experiment, args)
    for spec in specs:
        output_dir = args.workdir / spec.output_subdir
        command = build_train_command(spec, output_dir, args.python_bin)
        print(f"[{spec.name}] {spec.description}")
        print(format_command(command))
        print()


def run_experiments(args: argparse.Namespace) -> None:
    env = build_env(args.workdir, args.hf_endpoint)
    specs = specs_for_name(args.experiment, args)
    args.workdir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        output_dir = args.workdir / spec.output_subdir
        command = build_train_command(spec, output_dir, args.python_bin)
        prepare_output_dir_for_launch(output_dir)
        write_manifest(output_dir, spec, command, env)
        print(f"[{spec.name}] {spec.description}")
        print(f"Output: {output_dir}")
        print(format_command(command))
        print()
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    subparsers.add_parser("list", help="List the supported experiment names.")

    show_parser = subparsers.add_parser("show", help="Print the train command(s) for an experiment.")
    add_common_run_args(show_parser)

    run_parser = subparsers.add_parser("run", help="Execute the train command(s) for an experiment.")
    add_common_run_args(run_parser)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subcommand == "list":
        list_experiments()
    elif args.subcommand == "show":
        show_experiments(args)
    elif args.subcommand == "run":
        run_experiments(args)
    else:
        raise ValueError(f"Unsupported subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
