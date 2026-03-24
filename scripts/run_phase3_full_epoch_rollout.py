#!/usr/bin/env python3
"""Run the fair full-epoch JORA follow-up across up to three GPUs.

This launcher replaces the short-horizon phase-2 queue with a cleaner schedule:

- 1-epoch MMLU-200 core sweep on seed 42
- 1-epoch MMLU-200 shape probe on seed 42
- 3-epoch three-seed claim anchors against LoRA-r1 / LoRA-r2
- 3-epoch appendix full-benchmark runs for non-claim JORA cores on seed 42

The script reuses the phase-1 selected JORA learning rates unless explicit
overrides are supplied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_PATH = REPO_ROOT / "scripts" / "single_gpu_bf16_plan.py"
ROLLOUT_PATH = REPO_ROOT / "scripts" / "run_three_gpu_rollout.py"
EVALUATOR_PATH = REPO_ROOT / "scripts" / "evaluate_reasoning_benchmarks.py"
DEFAULT_WORKDIR = REPO_ROOT / "formal_runs" / "three_gpu_bf16_phase3"
DEFAULT_PHASE1_WORKDIR = REPO_ROOT / "formal_runs" / "three_gpu_bf16"
CORE_SWEEP_OUTPUT_NAME = "mmlu_200.json"
SHAPE_PROBE_OUTPUT_NAME = "mmlu_200.json"
FULL_MMLU_OUTPUT_NAME = "mmlu.json"
FULL_PARTIAL_OUTPUT_NAME = "arc_gsm8k.json"
FULL_MERGED_OUTPUT_NAME = "benchmarks.json"
CLAIM_SEEDS = (42, 1337, 2026)
DEFAULT_PHASE3_GPUS = [0, 1, 2]
DEFAULT_GPU_FREE_MEMORY_MB = 16_000
DEFAULT_GPU_FREE_UTILIZATION = 100


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load_module("single_gpu_bf16_plan_phase3", LAUNCHER_PATH)
rollout = _load_module("run_three_gpu_rollout_phase3", ROLLOUT_PATH)

build_env = rollout.build_env
ensure_workdir = rollout.ensure_workdir
evaluate_adapter = rollout.evaluate_adapter
load_json = rollout.load_json
merge_full_benchmarks = rollout.merge_full_benchmarks
start_subprocess = rollout.start_subprocess
start_training_task = rollout.start_training_task
update_state_phase = rollout.update_state_phase
write_json = rollout.write_json
benchmark_result_complete = rollout.benchmark_result_complete


@dataclass(frozen=True)
class ShapeCandidate:
    tag: str
    s_l: int
    s_r: int
    k: int

    @property
    def family(self) -> str:
        return f"jora_{self.tag}"


@dataclass(frozen=True)
class CoreCandidate:
    tag: str
    core: str
    block_size: int | None = None
    lowrank_r: int | None = None
    lowrank_alpha: float | None = None
    zero_init_core: bool | None = True
    s_l: int = 32
    s_r: int = 32
    k: int = 32

    @property
    def family(self) -> str:
        if self.core == "selective_diag":
            return f"jora_s{self.s_l}_k{self.k}"
        if self.core == "diag":
            return "jora_diag"
        if self.core == "block":
            return f"jora_block_bs{self.block_size}"
        if self.core == "lowrank":
            return f"jora_lowrank_r{self.lowrank_r}"
        raise ValueError(f"Unsupported JORA core: {self.core}")


CORE_CANDIDATES = (
    CoreCandidate(tag="selective_diag_s32_k32", core="selective_diag", zero_init_core=None, s_l=32, s_r=32, k=32),
    CoreCandidate(tag="diag", core="diag", zero_init_core=True, s_l=32, s_r=32, k=32),
    CoreCandidate(tag="block_bs4", core="block", block_size=4, zero_init_core=True, s_l=32, s_r=32, k=32),
    CoreCandidate(tag="lowrank_r1", core="lowrank", lowrank_r=1, lowrank_alpha=1.0, zero_init_core=True, s_l=32, s_r=32, k=32),
)

APPENDIX_CORE_CANDIDATES = tuple(candidate for candidate in CORE_CANDIDATES if candidate.core != "selective_diag")

SHAPE_CANDIDATES = (
    ShapeCandidate(tag="s32_k32", s_l=32, s_r=32, k=32),
    ShapeCandidate(tag="s96_k16", s_l=96, s_r=96, k=16),
    ShapeCandidate(tag="s96_k32", s_l=96, s_r=96, k=32),
)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--phase1-workdir", type=Path, default=DEFAULT_PHASE1_WORKDIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--hf-endpoint")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=DEFAULT_PHASE3_GPUS,
        help="GPU ids eligible for phase-3 scheduling. Defaults to opportunistic use of GPUs 0/1/2.",
    )
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument(
        "--gpu-free-memory-mb",
        type=int,
        default=DEFAULT_GPU_FREE_MEMORY_MB,
        help="Minimum free VRAM (MiB) required before launching a phase-3 job on a GPU.",
    )
    parser.add_argument(
        "--gpu-free-utilization",
        type=int,
        default=DEFAULT_GPU_FREE_UTILIZATION,
        help="Maximum GPU utilization allowed for scheduling. Use 100 to effectively ignore utilization.",
    )
    parser.add_argument("--core-sweep-epochs", type=float, default=1.0)
    parser.add_argument("--core-sweep-mmlu-limit", type=int, default=200)
    parser.add_argument("--shape-probe-epochs", type=float, default=1.0)
    parser.add_argument("--shape-probe-mmlu-limit", type=int, default=200)
    parser.add_argument("--anchor-epochs", type=float, default=3.0)
    parser.add_argument("--appendix-core-epochs", type=float, default=3.0)
    parser.add_argument("--jora-lr-theta", type=float)
    parser.add_argument("--jora-lr-core", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.gpus = list(dict.fromkeys(args.gpus))
    if not args.gpus:
        raise ValueError("At least one GPU id must be provided with --gpus.")
    if (args.jora_lr_theta is None) != (args.jora_lr_core is None):
        raise ValueError("Override both --jora-lr-theta and --jora-lr-core together.")
    return args


def logging_interval(num_train_epochs: float) -> int:
    return 50 if float(num_train_epochs) <= 1.0 else 100


def query_gpu_states() -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    states: dict[int, dict[str, int]] = {}
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        index_str, total_str, used_str, free_str, utilization_str = [part.strip() for part in line.split(",")]
        states[int(index_str)] = {
            "memory_total_mb": int(total_str),
            "memory_used_mb": int(used_str),
            "memory_free_mb": int(free_str),
            "utilization": int(utilization_str),
        }
    return states


def gpu_is_eligible(state: dict[str, int], free_memory_mb: int, free_utilization: int) -> bool:
    utilization_limit = max(0, min(int(free_utilization), 100))
    return state["memory_free_mb"] >= int(free_memory_mb) and state["utilization"] <= utilization_limit


def format_gpu_summary(gpu_ids: list[int], states: dict[int, dict[str, int]]) -> str:
    parts: list[str] = []
    for gpu_id in gpu_ids:
        state = states.get(gpu_id)
        if state is None:
            parts.append(f"GPU{gpu_id}: unavailable")
            continue
        parts.append(
            f"GPU{gpu_id}: free={state['memory_free_mb']}MB used={state['memory_used_mb']}MB util={state['utilization']}%"
        )
    return ", ".join(parts)


def eligible_gpu_ids(gpu_ids: list[int], free_memory_mb: int, free_utilization: int, dry_run: bool) -> list[int]:
    if dry_run:
        return sorted(dict.fromkeys(gpu_ids))
    states = query_gpu_states()
    return [
        gpu_id
        for gpu_id in sorted(dict.fromkeys(gpu_ids))
        if gpu_id in states and gpu_is_eligible(states[gpu_id], free_memory_mb, free_utilization)
    ]


def shape_candidate_by_tag(tag: str) -> ShapeCandidate:
    for candidate in SHAPE_CANDIDATES:
        if candidate.tag == tag:
            return candidate
    raise KeyError(f"Unknown JORA shape tag: {tag}")


def resolve_jora_lrs(args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    if args.jora_lr_theta is not None and args.jora_lr_core is not None:
        return (
            args.jora_lr_theta,
            args.jora_lr_core,
            {
                "source": "cli_override",
                "jora_lr_theta": args.jora_lr_theta,
                "jora_lr_core": args.jora_lr_core,
            },
        )

    selected_path = args.phase1_workdir / "m1" / "selected_lr.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Could not find phase-1 selected LR file: {selected_path}")

    payload = load_json(selected_path)
    theta = float(payload["jora_lr_theta"])
    core = float(payload["jora_lr_core"])
    source = {
        "source": "phase1_selected_lr",
        "selected_lr_path": str(selected_path),
        "selection": payload,
    }
    return theta, core, source


def with_epoch_schedule(spec, *, num_train_epochs: float, save_total_limit: int):
    return replace(
        spec,
        max_steps=None,
        num_train_epochs=float(num_train_epochs),
        logging_steps=logging_interval(num_train_epochs),
        save_strategy="epoch",
        save_steps=None,
        save_total_limit=save_total_limit,
    )


def family_name_for_spec(spec) -> str:
    if spec.method == "jora":
        if spec.jora_core == "selective_diag":
            return f"jora_s{spec.jora_s_l}_k{spec.jora_k}"
        if spec.jora_core == "diag":
            return "jora_diag"
        if spec.jora_core == "block":
            return f"jora_block_bs{spec.jora_block_size or 4}"
        if spec.jora_core == "lowrank":
            return f"jora_lowrank_r{spec.jora_lowrank_r or 8}"
        raise ValueError(f"Unsupported JORA core: {spec.jora_core}")
    if spec.method == "lora":
        return f"lora_r{spec.lora_r}"
    raise ValueError(f"Unsupported method: {spec.method}")


def entry_from_result(spec, output_dir: Path, result_path: Path, gpu_id: int, dry_run: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "output_dir": str(output_dir),
        "mmlu_output": str(result_path),
        "gpu_id": gpu_id,
        "seed": spec.seed,
        "method": spec.method,
        "family": family_name_for_spec(spec),
    }
    if spec.method == "jora":
        entry["jora_core"] = spec.jora_core
        entry["jora_zero_init_core"] = spec.jora_zero_init_core
        entry["shape"] = {
            "tag": f"s{spec.jora_s_l}_k{spec.jora_k}",
            "s_l": spec.jora_s_l,
            "s_r": spec.jora_s_r,
            "k": spec.jora_k,
        }
        if spec.jora_block_size is not None:
            entry["jora_block_size"] = spec.jora_block_size
        if spec.jora_lowrank_r is not None:
            entry["jora_lowrank_r"] = spec.jora_lowrank_r
        if spec.jora_lowrank_alpha is not None:
            entry["jora_lowrank_alpha"] = spec.jora_lowrank_alpha
    if spec.method == "lora":
        entry["lora_r"] = spec.lora_r

    if dry_run or not result_path.exists():
        entry["mmlu_accuracy"] = 0.0
        entry["trainable_params"] = None
        return entry

    payload = load_json(result_path)
    entry["mmlu_accuracy"] = float(payload["benchmarks"]["mmlu"]["accuracy"])
    entry["trainable_params"] = payload.get("parameters", {}).get("trainable_params")
    return entry


def build_core_sweep_specs(best_theta: float, best_core: float, num_train_epochs: float) -> list[Any]:
    base_jora = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_jora"],
            seed=CLAIM_SEEDS[0],
            jora_lr_theta=best_theta,
            jora_lr_core=best_core,
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
    )
    specs: list[Any] = []
    for candidate in CORE_CANDIDATES:
        specs.append(
            replace(
                base_jora,
                name=f"core_sweep_{candidate.family}_seed{base_jora.seed}",
                description=f"Phase-3 core sweep for {candidate.tag} on seed {base_jora.seed}.",
                output_subdir=f"core_sweep/{candidate.family}_seed{base_jora.seed}",
                jora_core=candidate.core,
                jora_block_size=candidate.block_size,
                jora_lowrank_r=candidate.lowrank_r,
                jora_lowrank_alpha=candidate.lowrank_alpha,
                jora_zero_init_core=candidate.zero_init_core,
                jora_s_l=candidate.s_l,
                jora_s_r=candidate.s_r,
                jora_k=candidate.k,
            )
        )

    baseline_lora_r1 = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_lora_r1"],
            name="core_sweep_lora_r1_seed42",
            description="Phase-3 core sweep baseline LoRA-r1 on seed 42.",
            output_subdir="core_sweep/lora_r1_seed42",
            seed=CLAIM_SEEDS[0],
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
    )
    baseline_lora_r2 = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_lora_r2"],
            name="core_sweep_lora_r2_seed42",
            description="Phase-3 core sweep baseline LoRA-r2 on seed 42.",
            output_subdir="core_sweep/lora_r2_seed42",
            seed=CLAIM_SEEDS[0],
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
    )
    return [*specs, baseline_lora_r1, baseline_lora_r2]


def build_shape_probe_specs(best_theta: float, best_core: float, num_train_epochs: float) -> list[Any]:
    base = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_jora"],
            seed=CLAIM_SEEDS[0],
            jora_lr_theta=best_theta,
            jora_lr_core=best_core,
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
    )
    specs = []
    for candidate in SHAPE_CANDIDATES:
        specs.append(
            replace(
                base,
                name=f"shape_probe_{candidate.family}_seed{base.seed}",
                description=f"Phase-3 JORA shape probe for {candidate.tag} on seed {base.seed}.",
                output_subdir=f"shape_probe/{candidate.family}_seed{base.seed}",
                jora_s_l=candidate.s_l,
                jora_s_r=candidate.s_r,
                jora_k=candidate.k,
            )
        )
    return specs


def build_anchor_specs(selected_shape: ShapeCandidate, best_theta: float, best_core: float, seed: int, num_train_epochs: float) -> list[Any]:
    jora = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_jora"],
            name=f"anchor_{selected_shape.family}_seed{seed}",
            description=f"Phase-3 three-seed anchor for {selected_shape.tag} on seed {seed}.",
            output_subdir=f"anchors/{selected_shape.family}_seed{seed}",
            seed=seed,
            jora_s_l=selected_shape.s_l,
            jora_s_r=selected_shape.s_r,
            jora_k=selected_shape.k,
            jora_lr_theta=best_theta,
            jora_lr_core=best_core,
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
    )
    lora_r1 = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_lora_r1"],
            name=f"anchor_lora_r1_seed{seed}",
            description=f"Phase-3 three-seed LoRA-r1 anchor on seed {seed}.",
            output_subdir=f"anchors/lora_r1_seed{seed}",
            seed=seed,
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
    )
    lora_r2 = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_lora_r2"],
            name=f"anchor_lora_r2_seed{seed}",
            description=f"Phase-3 three-seed LoRA-r2 anchor on seed {seed}.",
            output_subdir=f"anchors/lora_r2_seed{seed}",
            seed=seed,
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
    )
    return [jora, lora_r1, lora_r2]


def build_appendix_core_specs(best_theta: float, best_core: float, num_train_epochs: float) -> list[Any]:
    base = with_epoch_schedule(
        replace(
            launcher._base_named_specs()["m2b_jora"],
            seed=CLAIM_SEEDS[0],
            jora_lr_theta=best_theta,
            jora_lr_core=best_core,
        ),
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
    )
    specs: list[Any] = []
    for candidate in APPENDIX_CORE_CANDIDATES:
        specs.append(
            replace(
                base,
                name=f"appendix_{candidate.family}_seed{base.seed}",
                description=f"Phase-3 appendix full benchmark for {candidate.tag} on seed {base.seed}.",
                output_subdir=f"appendix_cores/{candidate.family}_seed{base.seed}",
                jora_core=candidate.core,
                jora_block_size=candidate.block_size,
                jora_lowrank_r=candidate.lowrank_r,
                jora_lowrank_alpha=candidate.lowrank_alpha,
                jora_zero_init_core=candidate.zero_init_core,
                jora_s_l=candidate.s_l,
                jora_s_r=candidate.s_r,
                jora_k=candidate.k,
            )
        )
    return specs


def run_parallel_specs_with_mmlu_eval(
    specs: list[Any],
    gpu_ids: list[int],
    args: argparse.Namespace,
    output_name: str,
    limit_mmlu: int | None = None,
) -> dict[str, dict[str, Any]]:
    pending = list(specs)
    running: dict[int, tuple[Any, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}

    while pending or running:
        made_progress = False

        launchable_gpus = [
            gpu_id
            for gpu_id in eligible_gpu_ids(
                gpu_ids,
                args.gpu_free_memory_mb,
                args.gpu_free_utilization,
                args.dry_run,
            )
            if gpu_id not in running
        ]

        while pending and launchable_gpus:
            gpu_id = launchable_gpus.pop(0)
            spec = pending.pop(0)
            task = start_training_task(spec, gpu_id, args.workdir, args.python_bin, args.hf_endpoint, args.dry_run)
            made_progress = True
            if task.process is None:
                result_path = evaluate_adapter(
                    task.output_dir,
                    gpu_id,
                    args.workdir,
                    args.python_bin,
                    args.hf_endpoint,
                    benchmarks=["mmlu"],
                    output_name=output_name,
                    limit_mmlu=limit_mmlu,
                    dry_run=args.dry_run,
                )
                entries[spec.name] = entry_from_result(spec, task.output_dir, result_path, gpu_id, args.dry_run)
            else:
                running[gpu_id] = (spec, task)

        completed_gpus: list[int] = []
        for gpu_id, (spec, task) in list(running.items()):
            return_code = task.process.poll()
            if return_code is None:
                continue
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, task.process.args)
            result_path = evaluate_adapter(
                task.output_dir,
                gpu_id,
                args.workdir,
                args.python_bin,
                args.hf_endpoint,
                benchmarks=["mmlu"],
                output_name=output_name,
                limit_mmlu=limit_mmlu,
                dry_run=args.dry_run,
            )
            entries[spec.name] = entry_from_result(spec, task.output_dir, result_path, gpu_id, args.dry_run)
            completed_gpus.append(gpu_id)
            made_progress = True

        for gpu_id in completed_gpus:
            del running[gpu_id]

        if pending or running:
            if not made_progress:
                if not args.dry_run:
                    states = query_gpu_states()
                    print("Waiting for an eligible GPU: " + format_gpu_summary(gpu_ids, states), flush=True)
                time.sleep(min(args.poll_interval, 10))

    return entries


def build_shape_leaderboard(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    leaderboard: list[dict[str, Any]] = []
    for spec_name, entry in runs.items():
        shape = entry.get("shape")
        if shape is None:
            continue
        leaderboard.append(
            {
                "name": spec_name,
                "family": entry["family"],
                "seed": entry["seed"],
                "mmlu_accuracy": entry["mmlu_accuracy"],
                "trainable_params": entry["trainable_params"],
                "tag": shape["tag"],
                "s_l": shape["s_l"],
                "s_r": shape["s_r"],
                "k": shape["k"],
                "output_dir": entry["output_dir"],
                "mmlu_output": entry["mmlu_output"],
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -row["mmlu_accuracy"],
            float("inf") if row["trainable_params"] is None else row["trainable_params"],
            row["k"],
            row["s_l"],
            row["name"],
        )
    )
    return leaderboard


def build_core_sweep_leaderboard(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    leaderboard: list[dict[str, Any]] = []
    for spec_name, entry in runs.items():
        row = {
            "name": spec_name,
            "family": entry["family"],
            "method": entry["method"],
            "seed": entry["seed"],
            "mmlu_accuracy": entry["mmlu_accuracy"],
            "trainable_params": entry["trainable_params"],
            "output_dir": entry["output_dir"],
            "mmlu_output": entry["mmlu_output"],
        }
        if entry["method"] == "jora":
            row["jora_core"] = entry["jora_core"]
            row["shape"] = entry["shape"]
            if "jora_block_size" in entry:
                row["jora_block_size"] = entry["jora_block_size"]
            if "jora_lowrank_r" in entry:
                row["jora_lowrank_r"] = entry["jora_lowrank_r"]
            if "jora_lowrank_alpha" in entry:
                row["jora_lowrank_alpha"] = entry["jora_lowrank_alpha"]
        if entry["method"] == "lora":
            row["lora_r"] = entry["lora_r"]
        leaderboard.append(row)

    leaderboard.sort(
        key=lambda row: (
            -row["mmlu_accuracy"],
            float("inf") if row["trainable_params"] is None else row["trainable_params"],
            row["family"],
            row["name"],
        )
    )
    return leaderboard


def select_best_shape(leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    if not leaderboard:
        raise ValueError("Shape leaderboard is empty.")
    return leaderboard[0]


def run_core_sweep(args: argparse.Namespace, state: dict[str, Any], best_theta: float, best_core: float) -> dict[str, Any]:
    specs = build_core_sweep_specs(best_theta, best_core, args.core_sweep_epochs)
    runs = run_parallel_specs_with_mmlu_eval(
        specs,
        args.gpus,
        args,
        output_name=CORE_SWEEP_OUTPUT_NAME,
        limit_mmlu=args.core_sweep_mmlu_limit,
    )
    leaderboard = build_core_sweep_leaderboard(runs)
    phase_payload = {"runs": runs, "leaderboard": leaderboard}
    update_state_phase(state, "core_sweep", phase_payload, args.workdir)
    return phase_payload


def run_shape_probe(args: argparse.Namespace, state: dict[str, Any], best_theta: float, best_core: float) -> ShapeCandidate:
    specs = build_shape_probe_specs(best_theta, best_core, args.shape_probe_epochs)
    runs = run_parallel_specs_with_mmlu_eval(
        specs,
        args.gpus,
        args,
        output_name=SHAPE_PROBE_OUTPUT_NAME,
        limit_mmlu=args.shape_probe_mmlu_limit,
    )
    leaderboard = build_shape_leaderboard(runs)
    selected = select_best_shape(leaderboard)
    write_json(args.workdir / "shape_probe" / "selected_shape.json", selected)
    phase_payload = {"runs": runs, "leaderboard": leaderboard, "selected": selected}
    update_state_phase(state, "shape_probe", phase_payload, args.workdir)
    return shape_candidate_by_tag(selected["tag"])


def run_parallel_full_evals(
    runs: dict[str, dict[str, Any]],
    gpu_ids: list[int],
    args: argparse.Namespace,
    *,
    mmlu_output_name: str,
) -> dict[str, str]:
    if args.dry_run:
        return {spec_name: "dry-run" for spec_name in runs}

    pending: list[tuple[str, Path]] = []
    outputs: dict[str, str] = {}
    for spec_name, entry in runs.items():
        output_dir = Path(entry["output_dir"])
        merged_path = output_dir / FULL_MERGED_OUTPUT_NAME
        if benchmark_result_complete(merged_path, ["mmlu", "arc_challenge", "gsm8k"]):
            outputs[spec_name] = str(merged_path)
            continue
        pending.append((spec_name, output_dir))

    running: dict[int, tuple[str, Path, subprocess.Popen[Any]]] = {}
    while pending or running:
        made_progress = False

        launchable_gpus = [
            gpu_id
            for gpu_id in eligible_gpu_ids(
                gpu_ids,
                args.gpu_free_memory_mb,
                args.gpu_free_utilization,
                args.dry_run,
            )
            if gpu_id not in running
        ]

        while pending and launchable_gpus:
            spec_name, output_dir = pending.pop(0)
            gpu_id = launchable_gpus.pop(0)
            partial_output = output_dir / FULL_PARTIAL_OUTPUT_NAME
            merged_output = output_dir / FULL_MERGED_OUTPUT_NAME
            if benchmark_result_complete(partial_output, ["arc_challenge", "gsm8k"]):
                merge_full_benchmarks(output_dir / mmlu_output_name, partial_output, merged_output)
                outputs[spec_name] = str(merged_output)
                made_progress = True
                continue

            env = build_env(args.workdir, args.hf_endpoint, gpu_id)
            command = [
                args.python_bin,
                str(EVALUATOR_PATH),
                "--adapter-path",
                str(output_dir),
                "--benchmarks",
                "arc_challenge",
                "gsm8k",
                "--output",
                str(partial_output),
            ]
            process = start_subprocess(command, env=env, dry_run=False)
            if process is None:
                raise RuntimeError("Full benchmark evaluation unexpectedly returned no process.")
            running[gpu_id] = (spec_name, output_dir, process)
            made_progress = True

        completed_gpus: list[int] = []
        for gpu_id, (spec_name, output_dir, process) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args)
            merged_output = output_dir / FULL_MERGED_OUTPUT_NAME
            merge_full_benchmarks(output_dir / mmlu_output_name, output_dir / FULL_PARTIAL_OUTPUT_NAME, merged_output)
            outputs[spec_name] = str(merged_output)
            completed_gpus.append(gpu_id)
            made_progress = True

        for gpu_id in completed_gpus:
            del running[gpu_id]

        if pending or running:
            if not made_progress:
                states = query_gpu_states()
                print("Waiting for an eligible GPU for full evals: " + format_gpu_summary(gpu_ids, states), flush=True)
                time.sleep(min(args.poll_interval, 10))

    return outputs


def run_anchor_rounds(
    args: argparse.Namespace,
    state: dict[str, Any],
    selected_shape: ShapeCandidate,
    best_theta: float,
    best_core: float,
) -> dict[str, Any]:
    phase_payload: dict[str, Any] = {
        "selected_shape": asdict(selected_shape),
        "seeds": list(CLAIM_SEEDS),
        "rounds": {},
    }
    for seed in CLAIM_SEEDS:
        specs = build_anchor_specs(selected_shape, best_theta, best_core, seed, args.anchor_epochs)
        runs = run_parallel_specs_with_mmlu_eval(specs, args.gpus, args, output_name=FULL_MMLU_OUTPUT_NAME)
        full_eval_outputs = run_parallel_full_evals(runs, args.gpus, args, mmlu_output_name=FULL_MMLU_OUTPUT_NAME)
        for spec_name, output_path in full_eval_outputs.items():
            runs[spec_name]["benchmarks_output"] = output_path
        phase_payload["rounds"][f"seed{seed}"] = {"seed": seed, "runs": runs}
        update_state_phase(state, "anchors", phase_payload, args.workdir)
    return phase_payload


def run_appendix_cores(args: argparse.Namespace, state: dict[str, Any], best_theta: float, best_core: float) -> dict[str, Any]:
    specs = build_appendix_core_specs(best_theta, best_core, args.appendix_core_epochs)
    runs = run_parallel_specs_with_mmlu_eval(specs, args.gpus, args, output_name=FULL_MMLU_OUTPUT_NAME)
    full_eval_outputs = run_parallel_full_evals(runs, args.gpus, args, mmlu_output_name=FULL_MMLU_OUTPUT_NAME)
    for spec_name, output_path in full_eval_outputs.items():
        runs[spec_name]["benchmarks_output"] = output_path
    phase_payload = {"seed": CLAIM_SEEDS[0], "runs": runs}
    update_state_phase(state, "appendix_cores", phase_payload, args.workdir)
    return phase_payload


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "values": [], "mean": None, "stdev": None}
    mean_value = statistics.mean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "values": values, "mean": mean_value, "stdev": stdev_value}


def aggregate_anchor_results(anchor_payload: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for round_payload in anchor_payload.get("rounds", {}).values():
        for entry in round_payload.get("runs", {}).values():
            family = entry["family"]
            group = families.setdefault(
                family,
                {
                    "family": family,
                    "method": entry["method"],
                    "seeds": [],
                    "trainable_params": [],
                    "benchmarks": {},
                    "output_dirs": [],
                },
            )
            group["seeds"].append(entry["seed"])
            group["output_dirs"].append(entry["output_dir"])
            if entry.get("trainable_params") is not None:
                group["trainable_params"].append(int(entry["trainable_params"]))

            benchmarks_path = entry.get("benchmarks_output")
            if not benchmarks_path or benchmarks_path == "dry-run":
                continue
            payload = load_json(Path(benchmarks_path))
            for benchmark_name, benchmark in payload.get("benchmarks", {}).items():
                group["benchmarks"].setdefault(benchmark_name, []).append(float(benchmark["accuracy"]))
            if "average_accuracy" in payload:
                group["benchmarks"].setdefault("average_accuracy", []).append(float(payload["average_accuracy"]))

    return {
        "selected_shape": anchor_payload.get("selected_shape"),
        "seeds": list(CLAIM_SEEDS),
        "families": {
            family: {
                "family": family,
                "method": payload["method"],
                "seeds": sorted(payload["seeds"]),
                "output_dirs": payload["output_dirs"],
                "trainable_params": _summary_stats(payload["trainable_params"]),
                "benchmarks": {
                    benchmark_name: _summary_stats(values)
                    for benchmark_name, values in sorted(payload["benchmarks"].items())
                },
            }
            for family, payload in families.items()
        },
    }


def aggregate_flat_run_results(payload: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for entry in payload.get("runs", {}).values():
        family = entry["family"]
        group = families.setdefault(
            family,
            {
                "family": family,
                "method": entry["method"],
                "seeds": [],
                "trainable_params": [],
                "benchmarks": {},
                "output_dirs": [],
            },
        )
        group["seeds"].append(entry["seed"])
        group["output_dirs"].append(entry["output_dir"])
        if entry.get("trainable_params") is not None:
            group["trainable_params"].append(int(entry["trainable_params"]))
        benchmarks_path = entry.get("benchmarks_output")
        if not benchmarks_path or benchmarks_path == "dry-run":
            continue
        result = load_json(Path(benchmarks_path))
        for benchmark_name, benchmark in result.get("benchmarks", {}).items():
            group["benchmarks"].setdefault(benchmark_name, []).append(float(benchmark["accuracy"]))
        if "average_accuracy" in result:
            group["benchmarks"].setdefault("average_accuracy", []).append(float(result["average_accuracy"]))

    return {
        "families": {
            family: {
                "family": family,
                "method": group["method"],
                "seeds": sorted(group["seeds"]),
                "output_dirs": group["output_dirs"],
                "trainable_params": _summary_stats(group["trainable_params"]),
                "benchmarks": {
                    benchmark_name: _summary_stats(values)
                    for benchmark_name, values in sorted(group["benchmarks"].items())
                },
            }
            for family, group in families.items()
        }
    }


def write_phase3_summary(
    args: argparse.Namespace,
    state: dict[str, Any],
    core_payload: dict[str, Any],
    shape_payload: dict[str, Any],
    anchor_payload: dict[str, Any],
    appendix_payload: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "core_sweep": {"leaderboard": core_payload.get("leaderboard", [])},
        "shape_probe": {
            "leaderboard": shape_payload.get("leaderboard", []),
            "selected": shape_payload.get("selected"),
        },
        "anchors": aggregate_anchor_results(anchor_payload),
        "appendix_cores": aggregate_flat_run_results(appendix_payload),
    }
    write_json(args.workdir / "summary.json", summary)
    update_state_phase(state, "aggregate", summary, args.workdir)
    return summary


def main() -> None:
    args = build_args()
    ensure_workdir(args.workdir)
    best_theta, best_core, lr_source = resolve_jora_lrs(args)

    state: dict[str, Any] = {
        "settings": {
            "workdir": str(args.workdir),
            "phase1_workdir": str(args.phase1_workdir),
            "gpus": args.gpus,
            "poll_interval": args.poll_interval,
            "gpu_free_memory_mb": args.gpu_free_memory_mb,
            "gpu_free_utilization": args.gpu_free_utilization,
            "core_sweep_epochs": args.core_sweep_epochs,
            "core_sweep_mmlu_limit": args.core_sweep_mmlu_limit,
            "shape_probe_epochs": args.shape_probe_epochs,
            "shape_probe_mmlu_limit": args.shape_probe_mmlu_limit,
            "anchor_epochs": args.anchor_epochs,
            "appendix_core_epochs": args.appendix_core_epochs,
            "dry_run": args.dry_run,
        },
        "lr_source": lr_source,
    }
    write_json(args.workdir / "rollout_state.json", state)

    core_payload = run_core_sweep(args, state, best_theta, best_core)
    selected_shape = run_shape_probe(args, state, best_theta, best_core)
    shape_payload = state["phases"]["shape_probe"]
    anchor_payload = run_anchor_rounds(args, state, selected_shape, best_theta, best_core)
    appendix_payload = run_appendix_cores(args, state, best_theta, best_core)
    write_phase3_summary(args, state, core_payload, shape_payload, anchor_payload, appendix_payload)


if __name__ == "__main__":
    main()
