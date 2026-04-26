#!/usr/bin/env python3
"""Phase B: selective_diag 5-run mainline experiment.

根据砍掉 Fisher/Refresh 后的执行计划：
- Step 4: selective_diag 5-run 主线实验

实验矩阵：
  Phase A: 4 个主线 run (s96/k16, 使用 paper-path 配置)
    - A1: selective_diag @ s96/k16, seed=42
    - A2: selective_diag @ s96/k16, seed=1337
    - A3: selective_diag @ s96/k16, seed=2026
    - A4: LoRA-r1 baseline (matched trainable budget)

  Phase B: diag-only-selected (frozen support from full JORA)
    - B1: selective_diag @ s96/k16, frozen support, seed=42

Phase A 需要使用 HF_ENDPOINT=hf-mirror.com
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "examples" / "sft" / "train.py"
DEFAULT_WORKDIR = REPO_ROOT / "formal_runs" / "selective_diag_5run"
MISTRAL_7B_MODEL_ID = "mistralai/Mistral-7B-v0.1"
LOCAL_MISTRAL_7B_PATH = Path("/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1")
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def _local_model_name_or_path(local_path: Path, fallback_model_id: str) -> str:
    if local_path.exists():
        return str(local_path)
    return fallback_model_id


def default_mistral_model_name_or_path() -> str:
    return _local_model_name_or_path(LOCAL_MISTRAL_7B_PATH, MISTRAL_7B_MODEL_ID)


@dataclass(frozen=True)
class TrainSpec:
    name: str
    description: str
    method: str  # "jora" or "lora"
    output_subdir: str
    seed: int
    jora_core: str = "selective_diag"
    jora_s_l: int = 96
    jora_s_r: int = 96
    jora_k: int = 16
    jora_lr_theta: float = 5e-3
    jora_lr_core: float = 1e-3
    jora_t_stat: int = 100
    jora_pairs_freeze_after_warmup: bool = True
    jora_magnitude: str = "none"
    lora_r: int = 1
    lora_alpha: int = 2
    max_steps: int = 2000
    num_train_epochs: float | None = None


def build_phase_a_specs() -> list[TrainSpec]:
    """Phase A: 4 mainline runs with paper-path selective_diag."""
    seeds = [42, 1337, 2026]
    specs = []
    for seed in seeds:
        specs.append(TrainSpec(
            name=f"selective_diag_s96_k16_seed{seed}",
            description=f"SelectiveDiagCore @ s96/k16, seed={seed}",
            method="jora",
            output_subdir=f"phase_a/selective_diag_s96_k16_seed{seed}",
            seed=seed,
        ))
    # LoRA-r1 baseline
    specs.append(TrainSpec(
        name="lora_r1_baseline_seed42",
        description="LoRA-r1 baseline (matched trainable budget)",
        method="lora",
        output_subdir="phase_a/lora_r1_seed42",
        seed=42,
    ))
    return specs


def build_phase_b_specs() -> list[TrainSpec]:
    """Phase B: diag-only-selected with frozen support from full JORA."""
    return [
        TrainSpec(
            name="diag_only_frozen_seed42",
            description="Diag-only-selected (frozen support from full JORA warmup), seed=42",
            method="jora",
            output_subdir="phase_b/diag_only_frozen_seed42",
            seed=42,
        ),
    ]


def build_train_command(spec: TrainSpec, output_dir: Path, python_bin: str, target_modules: str = "q_proj,o_proj") -> list[str]:
    model_path = default_mistral_model_name_or_path()
    
    command = [
        python_bin,
        str(TRAIN_SCRIPT),
        "--seed", str(spec.seed),
        "--model_name_or_path", model_path,
        "--dataset_name", "yahma/alpaca-cleaned",
        "--chat_template_format", "none",
        "--add_special_tokens", "False",
        "--append_concat_token", "False",
        "--splits", "train",
        "--torch_dtype", "bfloat16",
        "--logging_steps", "50",
        "--eval_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "500",
        "--save_total_limit", "2",
        "--report_to", "none",
        "--output_dir", str(output_dir),
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--max_seq_length", "512",
        "--dataset_text_field", "text",
        "--gradient_checkpointing", "True",
        "--use_reentrant", "False",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--max_steps", str(spec.max_steps),
        "--ddp_find_unused_parameters", "True",
    ]

    if spec.method == "jora":
        command.extend([
            "--use_peft_jora", "True",
            "--jora_core", spec.jora_core,
            "--jora_target_modules", target_modules,
            "--jora_magnitude", spec.jora_magnitude,
            "--jora_t_stat", str(spec.jora_t_stat),
            "--jora_pairs_freeze_after_warmup", "True" if spec.jora_pairs_freeze_after_warmup else "False",
            "--jora_selection_type", "topk_ema",
            "--jora_s_l", str(spec.jora_s_l),
            "--jora_s_r", str(spec.jora_s_r),
            "--jora_k", str(spec.jora_k),
            "--jora_lr_theta", f"{spec.jora_lr_theta:g}",
            "--jora_lr_core", f"{spec.jora_lr_core:g}",
        ])
    elif spec.method == "lora":
        command.extend([
            "--use_peft_lora", "True",
            "--lora_target_modules", target_modules,
            "--lora_r", str(spec.lora_r),
            "--lora_alpha", str(spec.lora_alpha),
            "--lora_dropout", "0.0",
        ])

    return command


def build_env(workdir: Path, hf_endpoint: str | None = None) -> dict[str, str]:
    import os
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Will be overridden by caller
    env["HF_HOME"] = str(workdir / "hf_home")
    env["HF_DATASETS_CACHE"] = str(workdir / "hf_datasets")
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_DATASETS_OFFLINE"] = "0"  # Allow online download for datasets
    env["HF_HUB_OFFLINE"] = "0"
    env["TRANSFORMERS_OFFLINE"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MPLCONFIGDIR"] = str(workdir / "mplconfig")
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_spec(spec: TrainSpec, workdir: Path, python_bin: str, hf_endpoint: str | None, gpu_id: int) -> dict:
    output_dir = workdir / spec.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = build_train_command(spec, output_dir, python_bin)
    env = build_env(workdir, hf_endpoint)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"\n{'='*70}")
    print(f"Running: {spec.name}")
    print(f"Description: {spec.description}")
    print(f"Output: {output_dir}")
    print(f"GPU: {gpu_id}")
    print(f"Command: {' '.join(command)}")
    print(f"HF_ENDPOINT: {env.get('HF_ENDPOINT', 'default')}")
    print(f"{'='*70}\n")
    
    # Write run manifest
    manifest = {
        "spec": asdict(spec),
        "command": command,
        "gpu_id": gpu_id,
        "hf_endpoint": hf_endpoint,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    
    # Run training
    result = subprocess.run(command, cwd=REPO_ROOT, env=env)
    
    return {
        "spec_name": spec.name,
        "output_dir": str(output_dir),
        "return_code": result.returncode,
        "success": result.returncode == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run selective_diag 5-run mainline experiment")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--hf-endpoint", default=HF_MIRROR_ENDPOINT,
                        help=f"HuggingFace endpoint (default: {HF_MIRROR_ENDPOINT})")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--phase", choices=["a", "b", "all"], default="all",
                        help="Which phase to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()
    
    args.workdir.mkdir(parents=True, exist_ok=True)
    
    specs = []
    if args.phase in ("a", "all"):
        specs.extend(build_phase_a_specs())
    if args.phase in ("b", "all"):
        specs.extend(build_phase_b_specs())
    
    print(f"\n{'#'*70}")
    print(f"SelectiveDiag 5-run Mainline Experiment")
    print(f"{'#'*70}")
    print(f"Workdir: {args.workdir}")
    print(f"GPU: {args.gpu_id}")
    print(f"HF Endpoint: {args.hf_endpoint}")
    print(f"Phase: {args.phase}")
    print(f"Specs to run: {len(specs)}")
    for spec in specs:
        print(f"  - {spec.name}: {spec.description}")
    print(f"{'#'*70}\n")
    
    if args.dry_run:
        print("Dry run - showing commands only\n")
        for spec in specs:
            command = build_train_command(spec, args.workdir / spec.output_subdir, args.python_bin)
            print(f"[{spec.name}]")
            print(f"  {' '.join(command)}\n")
        return
    
    results = []
    for spec in specs:
        result = run_spec(spec, args.workdir, args.python_bin, args.hf_endpoint, args.gpu_id)
        results.append(result)
        
        if not result["success"]:
            print(f"\n!!! FAILED: {spec.name} (return code: {result['return_code']})")
            response = input("Continue with next spec? [y/N] ")
            if response.lower() != 'y':
                print("Aborting.")
                break
    
    # Summary
    print(f"\n{'#'*70}")
    print("Experiment Summary")
    print(f"{'#'*70}")
    successes = [r for r in results if r["success"]]
    print(f"Total: {len(results)}, Success: {len(successes)}, Failed: {len(results) - len(successes)}")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['spec_name']}")
    
    # Save results
    results_path = args.workdir / "run_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
