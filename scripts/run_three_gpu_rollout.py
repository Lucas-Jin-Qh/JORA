#!/usr/bin/env python3
"""Orchestrate the formal bf16 JORA experiment rollout across up to three GPUs.

The rollout follows the agreed execution policy:

- Start immediately on one free GPU for M0 and M1.
- Wait for the other GPUs to become free before starting M2a.
- Run M2b wave 1 (JORA / LoRA-r1 / LoRA-r2) in parallel.
- Launch fixed-slot JORA only if the MMLU gate passes.
- Reuse the MMLU gate result and only run ARC-Challenge + GSM8K afterward.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_PATH = REPO_ROOT / "scripts" / "single_gpu_bf16_plan.py"
EVALUATOR_PATH = REPO_ROOT / "scripts" / "evaluate_reasoning_benchmarks.py"
DEFAULT_WORKDIR = REPO_ROOT / "formal_runs" / "three_gpu_bf16"
M0_OUTPUT_NAME = "mmlu.json"
M1_OUTPUT_NAME = "mmlu_200.json"
M2_OUTPUT_NAME = "mmlu.json"
FULL_PARTIAL_OUTPUT_NAME = "arc_gsm8k.json"
FULL_MERGED_OUTPUT_NAME = "benchmarks.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load_module("single_gpu_bf16_plan_rollout", LAUNCHER_PATH)


@dataclass
class RunningTask:
    spec_name: str
    output_dir: Path
    gpu_id: int
    process: subprocess.Popen[Any] | None


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--hf-endpoint")
    parser.add_argument("--start-gpu", type=int, default=0, help="GPU used for the immediate M0/M1 start.")
    parser.add_argument(
        "--serial-gpu",
        type=int,
        help="If set, run M2a and M2b serially on this single GPU instead of waiting for more GPUs.",
    )
    parser.add_argument(
        "--wait-gpus",
        type=int,
        nargs="+",
        default=[1, 2],
        help="GPUs that must become free before starting M2a.",
    )
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--gpu-free-memory-mb", type=int, default=1024)
    parser.add_argument("--gpu-free-utilization", type=int, default=10)
    parser.add_argument(
        "--fixed-slot-gate-margin",
        type=float,
        default=0.01,
        help="Launch fixed-slot if JORA MMLU is within this absolute accuracy margin of LoRA-r1.",
    )
    parser.add_argument("--m1-mmlu-limit", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_workdir(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)


def build_env(workdir: Path, hf_endpoint: str | None, gpu_id: int) -> dict[str, str]:
    env = launcher.build_env(workdir, hf_endpoint)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("MPLCONFIGDIR", str(workdir / "mplconfig"))
    return env


def run_subprocess(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def start_subprocess(command: list[str], env: dict[str, str], dry_run: bool) -> subprocess.Popen | None:
    print(" ".join(command), flush=True)
    if dry_run:
        return None
    return subprocess.Popen(command, cwd=REPO_ROOT, env=env)


def is_training_complete(output_dir: Path) -> bool:
    return (output_dir / "adapter_config.json").exists() and (output_dir / "adapter_model.safetensors").exists()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_result_complete(path: Path, benchmarks: list[str]) -> bool:
    if not path.exists():
        return False
    payload = load_json(path)
    existing = set(payload.get("benchmarks", {}))
    return all(benchmark in existing for benchmark in benchmarks)


def evaluate_adapter(
    output_dir: Path,
    gpu_id: int,
    workdir: Path,
    python_bin: str,
    hf_endpoint: str | None,
    benchmarks: list[str],
    output_name: str,
    limit_mmlu: int | None = None,
    limit_arc: int | None = None,
    limit_gsm8k: int | None = None,
    dry_run: bool = False,
) -> Path:
    output_path = output_dir / output_name
    if benchmark_result_complete(output_path, benchmarks):
        return output_path

    env = build_env(workdir, hf_endpoint, gpu_id)
    command = [
        python_bin,
        str(EVALUATOR_PATH),
        "--adapter-path",
        str(output_dir),
        "--benchmarks",
        *benchmarks,
        "--output",
        str(output_path),
    ]
    if limit_mmlu is not None:
        command.extend(["--limit-mmlu", str(limit_mmlu)])
    if limit_arc is not None:
        command.extend(["--limit-arc-challenge", str(limit_arc)])
    if limit_gsm8k is not None:
        command.extend(["--limit-gsm8k", str(limit_gsm8k)])
    run_subprocess(command, env=env, dry_run=dry_run)
    return output_path


def train_spec(spec, gpu_id: int, workdir: Path, python_bin: str, hf_endpoint: str | None, dry_run: bool) -> Path:
    output_dir = workdir / spec.output_subdir
    if is_training_complete(output_dir):
        return output_dir
    launcher.prepare_output_dir_for_launch(output_dir)

    env = build_env(workdir, hf_endpoint, gpu_id)
    command = launcher.build_train_command(spec, output_dir, python_bin)
    launcher.write_manifest(output_dir, spec, command, env)
    run_subprocess(command, env=env, dry_run=dry_run)
    return output_dir


def start_training_task(spec, gpu_id: int, workdir: Path, python_bin: str, hf_endpoint: str | None, dry_run: bool) -> RunningTask:
    output_dir = workdir / spec.output_subdir
    if is_training_complete(output_dir):
        return RunningTask(spec.name, output_dir, gpu_id, process=None)
    launcher.prepare_output_dir_for_launch(output_dir)

    env = build_env(workdir, hf_endpoint, gpu_id)
    command = launcher.build_train_command(spec, output_dir, python_bin)
    launcher.write_manifest(output_dir, spec, command, env)
    process = start_subprocess(command, env=env, dry_run=dry_run)
    return RunningTask(spec.name, output_dir, gpu_id, process=process)


def query_gpu_states() -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    states: dict[int, dict[str, int]] = {}
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        index_str, memory_used_str, utilization_str = [part.strip() for part in line.split(",")]
        states[int(index_str)] = {
            "memory_used_mb": int(memory_used_str),
            "utilization": int(utilization_str),
        }
    return states


def gpu_is_free(state: dict[str, int], free_memory_mb: int, free_utilization: int) -> bool:
    return state["memory_used_mb"] <= free_memory_mb and state["utilization"] <= free_utilization


def rollout_gpu_ids(start_gpu: int, wait_gpus: list[int], required_count: int) -> list[int]:
    ordered: list[int] = []
    for gpu_id in [start_gpu, *wait_gpus]:
        if gpu_id not in ordered:
            ordered.append(gpu_id)
    if len(ordered) < required_count:
        raise ValueError(
            f"Need at least {required_count} unique GPUs from --start-gpu/--wait-gpus, got {ordered}."
        )
    return ordered[:required_count]


def wait_for_gpus(gpu_ids: list[int], poll_interval: int, free_memory_mb: int, free_utilization: int, dry_run: bool) -> None:
    if dry_run:
        return
    while True:
        states = query_gpu_states()
        if all(gpu_is_free(states[gpu_id], free_memory_mb, free_utilization) for gpu_id in gpu_ids):
            return
        summary = ", ".join(
            f"GPU{gpu_id}: {states[gpu_id]['memory_used_mb']}MB/{states[gpu_id]['utilization']}%"
            for gpu_id in gpu_ids
        )
        print(f"Waiting for GPUs to free: {summary}", flush=True)
        time.sleep(poll_interval)


def mmlu_accuracy(result_path: Path) -> float:
    payload = load_json(result_path)
    return float(payload["benchmarks"]["mmlu"]["accuracy"])


def choose_best_m1_spec(m1_specs: list[Any], workdir: Path) -> tuple[Any, list[dict[str, Any]]]:
    leaderboard: list[dict[str, Any]] = []
    for spec in m1_specs:
        output_dir = workdir / spec.output_subdir
        result_path = output_dir / M1_OUTPUT_NAME
        payload = load_json(result_path)
        accuracy = float(payload["benchmarks"]["mmlu"]["accuracy"])
        leaderboard.append(
            {
                "name": spec.name,
                "accuracy": accuracy,
                "jora_lr_theta": spec.jora_lr_theta,
                "jora_lr_core": spec.jora_lr_core,
                "output_dir": str(output_dir),
            }
        )

    leaderboard.sort(key=lambda row: (-row["accuracy"], row["jora_lr_theta"], row["jora_lr_core"]))
    best = leaderboard[0]
    for spec in m1_specs:
        if spec.name == best["name"]:
            return spec, leaderboard
    raise RuntimeError(f"Could not resolve best M1 spec from leaderboard: {best}")


def should_launch_fixed_slot(jora_accuracy: float, lora_r1_accuracy: float, gate_margin: float) -> bool:
    return jora_accuracy + gate_margin >= lora_r1_accuracy


def merge_full_benchmarks(mmlu_path: Path, partial_path: Path, merged_path: Path) -> None:
    mmlu_payload = load_json(mmlu_path)
    partial_payload = load_json(partial_path)

    merged = {
        "model_name_or_path": partial_payload.get("model_name_or_path", mmlu_payload.get("model_name_or_path")),
        "adapter_path": partial_payload.get("adapter_path", mmlu_payload.get("adapter_path")),
        "parameters": partial_payload.get("parameters", mmlu_payload.get("parameters")),
        "benchmarks": {
            "mmlu": mmlu_payload["benchmarks"]["mmlu"],
        },
    }
    for benchmark_name, benchmark_value in partial_payload.get("benchmarks", {}).items():
        merged["benchmarks"][benchmark_name] = benchmark_value
    accuracies = [entry["accuracy"] for entry in merged["benchmarks"].values()]
    merged["average_accuracy"] = sum(accuracies) / len(accuracies)
    write_json(merged_path, merged)


def write_rollout_state(workdir: Path, state: dict[str, Any]) -> None:
    write_json(workdir / "rollout_state.json", state)


def update_state_phase(state: dict[str, Any], phase: str, payload: dict[str, Any], workdir: Path) -> None:
    state.setdefault("phases", {})[phase] = payload
    write_rollout_state(workdir, state)


def run_serial_specs_with_mmlu_eval(
    specs: list[Any],
    gpu_id: int,
    output_name: str,
    args: argparse.Namespace,
) -> dict[str, dict[str, str]]:
    runs: dict[str, dict[str, str]] = {}
    for spec in specs:
        output_dir = train_spec(spec, gpu_id, args.workdir, args.python_bin, args.hf_endpoint, args.dry_run)
        result_path = evaluate_adapter(
            output_dir,
            gpu_id,
            args.workdir,
            args.python_bin,
            args.hf_endpoint,
            benchmarks=["mmlu"],
            output_name=output_name,
            dry_run=args.dry_run,
        )
        runs[spec.name] = {
            "output_dir": str(output_dir),
            "mmlu_output": str(result_path),
        }
    return runs


def record_m2b_result(
    phase_payload: dict[str, Any],
    spec_name: str,
    output_dir: Path,
    result_path: Path,
    gpu_id: int,
    fixed_slot_name: str,
    mmlu_results: dict[str, float],
    completed_outputs: dict[str, Path],
    dry_run: bool,
) -> None:
    completed_outputs[spec_name] = output_dir
    mmlu_results[spec_name] = 0.0 if dry_run else mmlu_accuracy(result_path)
    entry = {
        "output_dir": str(output_dir),
        "mmlu_output": str(result_path),
        "gpu_id": gpu_id,
    }
    if spec_name == fixed_slot_name:
        entry["status"] = "completed"
        phase_payload["fixed_slot"] = entry
    else:
        phase_payload["wave1"][spec_name] = entry


def ensure_m2b_gate(
    phase_payload: dict[str, Any],
    mmlu_results: dict[str, float],
    gate_margin: float,
) -> tuple[bool | None, bool]:
    if phase_payload.get("gate") is not None:
        return bool(phase_payload["gate"]["pass"]), False
    if "m2b_jora" not in mmlu_results or "m2b_lora_r1" not in mmlu_results:
        return None, False

    gate_passed = should_launch_fixed_slot(
        mmlu_results["m2b_jora"],
        mmlu_results["m2b_lora_r1"],
        gate_margin,
    )
    phase_payload["gate"] = {
        "jora_mmlu": mmlu_results["m2b_jora"],
        "lora_r1_mmlu": mmlu_results["m2b_lora_r1"],
        "margin": gate_margin,
        "pass": gate_passed,
    }
    if not gate_passed and phase_payload.get("fixed_slot") is None:
        phase_payload["fixed_slot"] = {
            "status": "skipped",
            "reason": "mmlu_gate_failed",
        }
    return gate_passed, True


def maybe_launch_fixed_slot(
    args: argparse.Namespace,
    phase_payload: dict[str, Any],
    state: dict[str, Any],
    fixed_slot_spec,
    running: dict[int, RunningTask],
    free_gpus: set[int],
    mmlu_results: dict[str, float],
    completed_outputs: dict[str, Path],
) -> bool:
    gate = phase_payload.get("gate")
    if not gate or not gate.get("pass"):
        return False

    fixed_slot_entry = phase_payload.get("fixed_slot")
    if isinstance(fixed_slot_entry, dict) and fixed_slot_entry.get("status") in {"running", "completed"}:
        return False
    if any(task.spec_name == fixed_slot_spec.name for task in running.values()):
        return False
    if not free_gpus:
        phase_payload["fixed_slot"] = {"status": "waiting_for_gpu"}
        return True

    launch_gpu = min(free_gpus)
    free_gpus.remove(launch_gpu)
    task = start_training_task(
        fixed_slot_spec,
        launch_gpu,
        args.workdir,
        args.python_bin,
        args.hf_endpoint,
        args.dry_run,
    )
    if task.process is None:
        result_path = evaluate_adapter(
            task.output_dir,
            launch_gpu,
            args.workdir,
            args.python_bin,
            args.hf_endpoint,
            benchmarks=["mmlu"],
            output_name=M2_OUTPUT_NAME,
            dry_run=args.dry_run,
        )
        record_m2b_result(
            phase_payload,
            task.spec_name,
            task.output_dir,
            result_path,
            launch_gpu,
            fixed_slot_spec.name,
            mmlu_results,
            completed_outputs,
            args.dry_run,
        )
        free_gpus.add(launch_gpu)
    else:
        phase_payload["fixed_slot"] = {
            "status": "running",
            "output_dir": str(task.output_dir),
            "gpu_id": launch_gpu,
        }
        running[launch_gpu] = task
    return True


def run_m2b_full_evals(
    args: argparse.Namespace,
    phase_payload: dict[str, Any],
    state: dict[str, Any],
    completed_outputs: dict[str, Path],
    accepted: list[str],
    gpu_ids: list[int],
) -> None:
    if args.dry_run:
        phase_payload["full_eval"] = {spec_name: "dry-run" for spec_name in accepted}
        update_state_phase(state, "m2b", phase_payload, args.workdir)
        return

    pending_eval: list[tuple[str, Path]] = []
    for spec_name in accepted:
        output_dir = completed_outputs[spec_name]
        full_output = output_dir / FULL_MERGED_OUTPUT_NAME
        if benchmark_result_complete(full_output, ["mmlu", "arc_challenge", "gsm8k"]):
            phase_payload["full_eval"][spec_name] = str(full_output)
            continue
        pending_eval.append((spec_name, output_dir))

    running_evals: dict[int, tuple[str, Path, subprocess.Popen[Any]]] = {}
    available_eval_gpus = set(gpu_ids)
    while pending_eval or running_evals:
        while pending_eval and available_eval_gpus:
            spec_name, output_dir = pending_eval.pop(0)
            gpu_id = min(available_eval_gpus)
            available_eval_gpus.remove(gpu_id)
            partial_output = output_dir / FULL_PARTIAL_OUTPUT_NAME
            if benchmark_result_complete(partial_output, ["arc_challenge", "gsm8k"]):
                merge_full_benchmarks(output_dir / M2_OUTPUT_NAME, partial_output, output_dir / FULL_MERGED_OUTPUT_NAME)
                phase_payload["full_eval"][spec_name] = str(output_dir / FULL_MERGED_OUTPUT_NAME)
                available_eval_gpus.add(gpu_id)
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
            running_evals[gpu_id] = (spec_name, output_dir, process)

        completed_eval_gpus: list[int] = []
        for gpu_id, (spec_name, output_dir, process) in list(running_evals.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args)
            merge_full_benchmarks(output_dir / M2_OUTPUT_NAME, output_dir / FULL_PARTIAL_OUTPUT_NAME, output_dir / FULL_MERGED_OUTPUT_NAME)
            phase_payload["full_eval"][spec_name] = str(output_dir / FULL_MERGED_OUTPUT_NAME)
            completed_eval_gpus.append(gpu_id)
            update_state_phase(state, "m2b", phase_payload, args.workdir)

        for gpu_id in completed_eval_gpus:
            del running_evals[gpu_id]
            available_eval_gpus.add(gpu_id)

        if running_evals:
            time.sleep(10)


def run_m0(args: argparse.Namespace, state: dict[str, Any]) -> None:
    phase_payload: dict[str, Any] = {"gpu": args.start_gpu, "runs": {}}
    for spec_name in ("m0_jora", "m0_lora_r1"):
        spec = launcher._base_named_specs()[spec_name]
        output_dir = train_spec(spec, args.start_gpu, args.workdir, args.python_bin, args.hf_endpoint, args.dry_run)
        result_path = evaluate_adapter(
            output_dir,
            args.start_gpu,
            args.workdir,
            args.python_bin,
            args.hf_endpoint,
            benchmarks=["mmlu"],
            output_name=M0_OUTPUT_NAME,
            dry_run=args.dry_run,
        )
        phase_payload["runs"][spec_name] = {
            "output_dir": str(output_dir),
            "mmlu_output": str(result_path),
        }
        update_state_phase(state, "m0", phase_payload, args.workdir)


def run_m1(args: argparse.Namespace, state: dict[str, Any]) -> tuple[float, float]:
    phase_payload: dict[str, Any] = {"gpu": args.start_gpu, "runs": {}, "selected": None}
    m1_specs = launcher._m1_sweep_specs()
    for spec in m1_specs:
        output_dir = train_spec(spec, args.start_gpu, args.workdir, args.python_bin, args.hf_endpoint, args.dry_run)
        result_path = evaluate_adapter(
            output_dir,
            args.start_gpu,
            args.workdir,
            args.python_bin,
            args.hf_endpoint,
            benchmarks=["mmlu"],
            output_name=M1_OUTPUT_NAME,
            limit_mmlu=args.m1_mmlu_limit,
            dry_run=args.dry_run,
        )
        phase_payload["runs"][spec.name] = {
            "output_dir": str(output_dir),
            "mmlu_output": str(result_path),
        }
        update_state_phase(state, "m1", phase_payload, args.workdir)

    if args.dry_run:
        return launcher.DEFAULT_JORA_LR_THETA, launcher.DEFAULT_JORA_LR_CORE

    best_spec, leaderboard = choose_best_m1_spec(m1_specs, args.workdir)
    selected = {
        "name": best_spec.name,
        "jora_lr_theta": best_spec.jora_lr_theta,
        "jora_lr_core": best_spec.jora_lr_core,
        "leaderboard": leaderboard,
    }
    write_json(args.workdir / "m1" / "selected_lr.json", selected)
    phase_payload["selected"] = selected
    update_state_phase(state, "m1", phase_payload, args.workdir)
    return best_spec.jora_lr_theta, best_spec.jora_lr_core


def run_m2a_serial(args: argparse.Namespace, state: dict[str, Any], best_theta: float, best_core: float) -> None:
    if args.serial_gpu is None:
        raise ValueError("run_m2a_serial requires --serial-gpu.")

    phase_payload: dict[str, Any] = {"gpu": args.serial_gpu, "mode": "serial", "runs": {}}
    specs = [
        replace(launcher._base_named_specs()["m2a_jora"], jora_lr_theta=best_theta, jora_lr_core=best_core),
        launcher._base_named_specs()["m2a_lora_r1"],
    ]
    phase_payload["runs"] = run_serial_specs_with_mmlu_eval(specs, args.serial_gpu, M2_OUTPUT_NAME, args)
    update_state_phase(state, "m2a", phase_payload, args.workdir)


def run_m2a(args: argparse.Namespace, state: dict[str, Any], best_theta: float, best_core: float) -> None:
    if args.serial_gpu is not None:
        run_m2a_serial(args, state, best_theta, best_core)
        return

    phase_payload: dict[str, Any] = {"runs": {}}
    wait_for_gpus(args.wait_gpus, args.poll_interval, args.gpu_free_memory_mb, args.gpu_free_utilization, args.dry_run)
    gpu_ids = rollout_gpu_ids(args.start_gpu, args.wait_gpus, required_count=2)

    tasks: list[tuple[Any, int]] = [
        (replace(launcher._base_named_specs()["m2a_jora"], jora_lr_theta=best_theta, jora_lr_core=best_core), gpu_ids[0]),
        (launcher._base_named_specs()["m2a_lora_r1"], gpu_ids[1]),
    ]

    running: list[RunningTask] = []
    for spec, gpu_id in tasks:
        task = start_training_task(spec, gpu_id, args.workdir, args.python_bin, args.hf_endpoint, args.dry_run)
        if task.process is None:
            result_path = evaluate_adapter(
                task.output_dir,
                gpu_id,
                args.workdir,
                args.python_bin,
                args.hf_endpoint,
                benchmarks=["mmlu"],
                output_name=M2_OUTPUT_NAME,
                dry_run=args.dry_run,
            )
            phase_payload["runs"][spec.name] = {
                "output_dir": str(task.output_dir),
                "mmlu_output": str(result_path),
            }
        else:
            running.append(task)

    while running:
        next_running: list[RunningTask] = []
        for task in running:
            return_code = task.process.poll()
            if return_code is None:
                next_running.append(task)
                continue
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, task.process.args)
            result_path = evaluate_adapter(
                task.output_dir,
                task.gpu_id,
                args.workdir,
                args.python_bin,
                args.hf_endpoint,
                benchmarks=["mmlu"],
                output_name=M2_OUTPUT_NAME,
                dry_run=args.dry_run,
            )
            phase_payload["runs"][task.spec_name] = {
                "output_dir": str(task.output_dir),
                "mmlu_output": str(result_path),
            }
            update_state_phase(state, "m2a", phase_payload, args.workdir)
        running = next_running
        if running:
            time.sleep(10)

    update_state_phase(state, "m2a", phase_payload, args.workdir)


def run_m2b_serial(args: argparse.Namespace, state: dict[str, Any], best_theta: float, best_core: float) -> None:
    if args.serial_gpu is None:
        raise ValueError("run_m2b_serial requires --serial-gpu.")

    phase_payload: dict[str, Any] = {
        "gpu": args.serial_gpu,
        "mode": "serial",
        "wave1": {},
        "fixed_slot": None,
        "full_eval": {},
        "gate": None,
    }
    wave1_specs = [
        replace(launcher._base_named_specs()["m2b_jora"], jora_lr_theta=best_theta, jora_lr_core=best_core),
        launcher._base_named_specs()["m2b_lora_r1"],
        launcher._base_named_specs()["m2b_lora_r2"],
    ]
    fixed_slot_spec = replace(
        launcher._base_named_specs()["m2b_fixed_jora"],
        jora_lr_theta=best_theta,
        jora_lr_core=best_core,
    )
    mmlu_results: dict[str, float] = {}
    completed_outputs: dict[str, Path] = {}

    for spec in wave1_specs:
        output_dir = train_spec(spec, args.serial_gpu, args.workdir, args.python_bin, args.hf_endpoint, args.dry_run)
        result_path = evaluate_adapter(
            output_dir,
            args.serial_gpu,
            args.workdir,
            args.python_bin,
            args.hf_endpoint,
            benchmarks=["mmlu"],
            output_name=M2_OUTPUT_NAME,
            dry_run=args.dry_run,
        )
        record_m2b_result(
            phase_payload,
            spec.name,
            output_dir,
            result_path,
            args.serial_gpu,
            fixed_slot_spec.name,
            mmlu_results,
            completed_outputs,
            args.dry_run,
        )
        update_state_phase(state, "m2b", phase_payload, args.workdir)

    gate_passed, gate_changed = ensure_m2b_gate(phase_payload, mmlu_results, args.fixed_slot_gate_margin)
    if gate_changed:
        update_state_phase(state, "m2b", phase_payload, args.workdir)
    if not gate_passed:
        return

    output_dir = train_spec(
        fixed_slot_spec,
        args.serial_gpu,
        args.workdir,
        args.python_bin,
        args.hf_endpoint,
        args.dry_run,
    )
    result_path = evaluate_adapter(
        output_dir,
        args.serial_gpu,
        args.workdir,
        args.python_bin,
        args.hf_endpoint,
        benchmarks=["mmlu"],
        output_name=M2_OUTPUT_NAME,
        dry_run=args.dry_run,
    )
    record_m2b_result(
        phase_payload,
        fixed_slot_spec.name,
        output_dir,
        result_path,
        args.serial_gpu,
        fixed_slot_spec.name,
        mmlu_results,
        completed_outputs,
        args.dry_run,
    )
    update_state_phase(state, "m2b", phase_payload, args.workdir)

    accepted = ["m2b_jora", "m2b_lora_r1", "m2b_lora_r2", "m2b_fixed_jora"]
    run_m2b_full_evals(args, phase_payload, state, completed_outputs, accepted, [args.serial_gpu])


def run_m2b(args: argparse.Namespace, state: dict[str, Any], best_theta: float, best_core: float) -> None:
    if args.serial_gpu is not None:
        run_m2b_serial(args, state, best_theta, best_core)
        return

    phase_payload: dict[str, Any] = {
        "wave1": {},
        "fixed_slot": None,
        "full_eval": {},
        "gate": None,
    }

    wave1_specs = [
        replace(launcher._base_named_specs()["m2b_jora"], jora_lr_theta=best_theta, jora_lr_core=best_core),
        launcher._base_named_specs()["m2b_lora_r1"],
        launcher._base_named_specs()["m2b_lora_r2"],
    ]
    gpu_ids = rollout_gpu_ids(args.start_gpu, args.wait_gpus, required_count=3)
    running: dict[int, RunningTask] = {}
    free_gpus: set[int] = set()
    mmlu_results: dict[str, float] = {}
    completed_outputs: dict[str, Path] = {}
    fixed_slot_spec = replace(
        launcher._base_named_specs()["m2b_fixed_jora"],
        jora_lr_theta=best_theta,
        jora_lr_core=best_core,
    )

    for spec, gpu_id in zip(wave1_specs, gpu_ids):
        task = start_training_task(spec, gpu_id, args.workdir, args.python_bin, args.hf_endpoint, args.dry_run)
        if task.process is None:
            result_path = evaluate_adapter(
                task.output_dir,
                gpu_id,
                args.workdir,
                args.python_bin,
                args.hf_endpoint,
                benchmarks=["mmlu"],
                output_name=M2_OUTPUT_NAME,
                dry_run=args.dry_run,
            )
            record_m2b_result(
                phase_payload,
                spec.name,
                task.output_dir,
                result_path,
                gpu_id,
                fixed_slot_spec.name,
                mmlu_results,
                completed_outputs,
                args.dry_run,
            )
            free_gpus.add(gpu_id)
        else:
            running[gpu_id] = task

    gate_passed, gate_changed = ensure_m2b_gate(phase_payload, mmlu_results, args.fixed_slot_gate_margin)
    launch_changed = maybe_launch_fixed_slot(
        args,
        phase_payload,
        state,
        fixed_slot_spec,
        running,
        free_gpus,
        mmlu_results,
        completed_outputs,
    )
    if phase_payload["wave1"] or gate_changed or launch_changed:
        update_state_phase(state, "m2b", phase_payload, args.workdir)

    while running:
        completed_gpu_ids: list[int] = []
        phase_changed = False
        for gpu_id, task in list(running.items()):
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
                output_name=M2_OUTPUT_NAME,
                dry_run=args.dry_run,
            )
            record_m2b_result(
                phase_payload,
                task.spec_name,
                task.output_dir,
                result_path,
                gpu_id,
                fixed_slot_spec.name,
                mmlu_results,
                completed_outputs,
                args.dry_run,
            )
            free_gpus.add(gpu_id)
            completed_gpu_ids.append(gpu_id)
            phase_changed = True

        for gpu_id in completed_gpu_ids:
            del running[gpu_id]

        gate_passed, gate_changed = ensure_m2b_gate(phase_payload, mmlu_results, args.fixed_slot_gate_margin)
        phase_changed = phase_changed or gate_changed
        if gate_passed:
            phase_changed = maybe_launch_fixed_slot(
                args,
                phase_payload,
                state,
                fixed_slot_spec,
                running,
                free_gpus,
                mmlu_results,
                completed_outputs,
            ) or phase_changed

        if phase_changed:
            update_state_phase(state, "m2b", phase_payload, args.workdir)

        if running:
            time.sleep(10)

    gate = phase_payload.get("gate")
    gate_passed = bool(gate and gate.get("pass"))
    if not gate_passed:
        update_state_phase(state, "m2b", phase_payload, args.workdir)
        return

    accepted = ["m2b_jora", "m2b_lora_r1", "m2b_lora_r2"]
    if phase_payload.get("fixed_slot") and phase_payload["fixed_slot"].get("status") == "completed":
        accepted.append("m2b_fixed_jora")
    accepted = list(dict.fromkeys(accepted))
    run_m2b_full_evals(args, phase_payload, state, completed_outputs, accepted, gpu_ids)


def main() -> None:
    args = build_args()
    ensure_workdir(args.workdir)
    state: dict[str, Any] = {
        "settings": {
            "workdir": str(args.workdir),
            "start_gpu": args.start_gpu,
            "serial_gpu": args.serial_gpu,
            "wait_gpus": args.wait_gpus,
            "poll_interval": args.poll_interval,
            "m1_mmlu_limit": args.m1_mmlu_limit,
            "fixed_slot_gate_margin": args.fixed_slot_gate_margin,
            "dry_run": args.dry_run,
        }
    }
    write_rollout_state(args.workdir, state)

    run_m0(args, state)
    best_theta, best_core = run_m1(args, state)
    run_m2a(args, state, best_theta, best_core)
    run_m2b(args, state, best_theta, best_core)


if __name__ == "__main__":
    main()
