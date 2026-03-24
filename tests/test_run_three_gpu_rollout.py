from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module(name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rollout = _load_script_module("run_three_gpu_rollout", "scripts/run_three_gpu_rollout.py")


class FakeProcess:
    def __init__(self, args):
        self.args = args

    def poll(self):
        return 0


def make_args(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    margin: float = 0.01,
    serial_gpu: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        workdir=tmp_path,
        python_bin=sys.executable,
        hf_endpoint=None,
        start_gpu=0,
        serial_gpu=serial_gpu,
        wait_gpus=[1, 2],
        poll_interval=1,
        gpu_free_memory_mb=1024,
        gpu_free_utilization=10,
        fixed_slot_gate_margin=margin,
        m1_mmlu_limit=200,
        dry_run=dry_run,
    )


def test_should_launch_fixed_slot_within_margin():
    assert rollout.should_launch_fixed_slot(0.299, 0.300, 0.01)
    assert not rollout.should_launch_fixed_slot(0.280, 0.300, 0.01)


def test_rollout_gpu_ids_deduplicates_and_validates_count():
    assert rollout.rollout_gpu_ids(2, [2, 1, 0], required_count=3) == [2, 1, 0]
    try:
        rollout.rollout_gpu_ids(0, [0, 1], required_count=3)
    except ValueError as exc:
        assert "Need at least 3 unique GPUs" in str(exc)
    else:
        raise AssertionError("Expected rollout_gpu_ids to reject insufficient unique GPUs.")


def test_training_complete_requires_adapter_files(tmp_path):
    assert not rollout.is_training_complete(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert not rollout.is_training_complete(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
    assert rollout.is_training_complete(tmp_path)


def test_build_env_keeps_shared_hf_cache_and_gpu_specific_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    env = rollout.build_env(tmp_path, hf_endpoint="https://hf-mirror.com", gpu_id=2)

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["HF_HOME"] == str(rollout.launcher.shared_hf_home_path())
    assert env["HF_DATASETS_CACHE"] == str(rollout.launcher.shared_hf_datasets_cache_path())
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert env["MPLCONFIGDIR"] == str(tmp_path / "mplconfig")


def test_choose_best_m1_prefers_accuracy_then_lower_lr(tmp_path):
    specs = rollout.launcher._m1_sweep_specs()[:3]
    accuracies = [0.31, 0.31, 0.29]
    for spec, accuracy in zip(specs, accuracies):
        output_dir = tmp_path / spec.output_subdir
        output_dir.mkdir(parents=True)
        payload = {
            "benchmarks": {
                "mmlu": {
                    "accuracy": accuracy,
                }
            }
        }
        (output_dir / rollout.M1_OUTPUT_NAME).write_text(__import__("json").dumps(payload), encoding="utf-8")

    best_spec, leaderboard = rollout.choose_best_m1_spec(specs, tmp_path)
    assert leaderboard[0]["accuracy"] == 0.31
    assert best_spec.jora_lr_theta == min(spec.jora_lr_theta for spec in specs[:2])


def test_train_spec_retries_manifest_only_output_dir(tmp_path, monkeypatch):
    spec = rollout.launcher._base_named_specs()["m2a_jora"]
    output_dir = tmp_path / spec.output_subdir
    output_dir.mkdir(parents=True)
    (output_dir / "run_spec.json").write_text("{}", encoding="utf-8")
    (output_dir / "run_command.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    seen_commands: list[list[str]] = []

    def fake_run_subprocess(command, env, dry_run):
        seen_commands.append(command)

    monkeypatch.setattr(rollout, "run_subprocess", fake_run_subprocess)

    returned_dir = rollout.train_spec(spec, 0, tmp_path, sys.executable, None, False)

    assert returned_dir == output_dir
    assert seen_commands
    assert (output_dir / "run_spec.json").exists()
    assert (output_dir / "run_command.sh").exists()


def test_run_m2a_serial_uses_single_gpu(tmp_path, monkeypatch):
    args = make_args(tmp_path, serial_gpu=0)
    state: dict[str, object] = {}
    seen_gpu_ids: list[int] = []
    output_to_spec: dict[Path, str] = {}

    def fail_wait_for_gpus(*_args, **_kwargs):
        raise AssertionError("Serial mode should not wait for extra GPUs.")

    def fake_train_spec(spec, gpu_id, workdir, python_bin, hf_endpoint, dry_run):
        seen_gpu_ids.append(gpu_id)
        output_dir = workdir / spec.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_to_spec[output_dir] = spec.name
        return output_dir

    def fake_evaluate_adapter(output_dir, gpu_id, workdir, python_bin, hf_endpoint, benchmarks, output_name, **kwargs):
        assert gpu_id == 0
        output_path = output_dir / output_name
        payload = {"benchmarks": {"mmlu": {"accuracy": 0.25}}}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return output_path

    monkeypatch.setattr(rollout, "wait_for_gpus", fail_wait_for_gpus)
    monkeypatch.setattr(rollout, "train_spec", fake_train_spec)
    monkeypatch.setattr(rollout, "evaluate_adapter", fake_evaluate_adapter)

    rollout.run_m2a(args, state, best_theta=1e-3, best_core=5e-4)

    assert seen_gpu_ids == [0, 0]
    phase = state["phases"]["m2a"]
    assert phase["gpu"] == 0
    assert phase["mode"] == "serial"
    assert set(phase["runs"]) == {"m2a_jora", "m2a_lora_r1"}


def test_run_m2b_resume_path_launches_fixed_slot_and_full_eval(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    state: dict[str, object] = {}
    output_to_spec: dict[Path, str] = {}
    mmlu_scores = {
        "m2b_jora": 0.300,
        "m2b_lora_r1": 0.305,
        "m2b_lora_r2": 0.315,
        "m2b_fixed_jora": 0.298,
    }

    def fake_start_training_task(spec, gpu_id, workdir, python_bin, hf_endpoint, dry_run):
        output_dir = workdir / spec.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_to_spec[output_dir] = spec.name
        return rollout.RunningTask(spec.name, output_dir, gpu_id, process=None)

    def fake_evaluate_adapter(output_dir, gpu_id, workdir, python_bin, hf_endpoint, benchmarks, output_name, **kwargs):
        output_path = output_dir / output_name
        spec_name = output_to_spec[output_dir]
        if benchmarks == ["mmlu"]:
            payload = {
                "benchmarks": {
                    "mmlu": {
                        "accuracy": mmlu_scores[spec_name],
                    }
                }
            }
        else:
            payload = {
                "model_name_or_path": "mistralai/Mistral-7B-v0.1",
                "adapter_path": str(output_dir),
                "parameters": {"trainable_params": 123},
                "benchmarks": {
                    "arc_challenge": {"accuracy": 0.20},
                    "gsm8k": {"accuracy": 0.10},
                },
            }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return output_path

    def fake_start_subprocess(command, env, dry_run):
        assert not dry_run
        output_path = Path(command[-1])
        payload = {
            "model_name_or_path": "mistralai/Mistral-7B-v0.1",
            "adapter_path": str(output_path.parent),
            "parameters": {"trainable_params": 123},
            "benchmarks": {
                "arc_challenge": {"accuracy": 0.20},
                "gsm8k": {"accuracy": 0.10},
            },
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return FakeProcess(command)

    monkeypatch.setattr(rollout, "start_training_task", fake_start_training_task)
    monkeypatch.setattr(rollout, "evaluate_adapter", fake_evaluate_adapter)
    monkeypatch.setattr(rollout, "start_subprocess", fake_start_subprocess)

    rollout.run_m2b(args, state, best_theta=5e-3, best_core=1e-3)

    phase = state["phases"]["m2b"]
    assert phase["gate"]["pass"] is True
    assert phase["fixed_slot"]["status"] == "completed"
    assert set(phase["wave1"]) == {"m2b_jora", "m2b_lora_r1", "m2b_lora_r2"}
    assert set(phase["full_eval"]) == {"m2b_jora", "m2b_lora_r1", "m2b_lora_r2", "m2b_fixed_jora"}
    merged_path = Path(phase["full_eval"]["m2b_jora"])
    merged_payload = json.loads(merged_path.read_text(encoding="utf-8"))
    assert set(merged_payload["benchmarks"]) == {"mmlu", "arc_challenge", "gsm8k"}


def test_run_m2b_gate_fail_skips_fixed_slot_and_full_eval(tmp_path, monkeypatch):
    args = make_args(tmp_path, margin=0.01)
    state: dict[str, object] = {}
    output_to_spec: dict[Path, str] = {}
    mmlu_scores = {
        "m2b_jora": 0.270,
        "m2b_lora_r1": 0.300,
        "m2b_lora_r2": 0.310,
    }

    def fake_start_training_task(spec, gpu_id, workdir, python_bin, hf_endpoint, dry_run):
        output_dir = workdir / spec.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_to_spec[output_dir] = spec.name
        return rollout.RunningTask(spec.name, output_dir, gpu_id, process=None)

    def fake_evaluate_adapter(output_dir, gpu_id, workdir, python_bin, hf_endpoint, benchmarks, output_name, **kwargs):
        assert benchmarks == ["mmlu"]
        output_path = output_dir / output_name
        payload = {
            "benchmarks": {
                "mmlu": {
                    "accuracy": mmlu_scores[output_to_spec[output_dir]],
                }
            }
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return output_path

    def fail_start_subprocess(command, env, dry_run):
        raise AssertionError("Full eval should not start when the MMLU gate fails.")

    monkeypatch.setattr(rollout, "start_training_task", fake_start_training_task)
    monkeypatch.setattr(rollout, "evaluate_adapter", fake_evaluate_adapter)
    monkeypatch.setattr(rollout, "start_subprocess", fail_start_subprocess)

    rollout.run_m2b(args, state, best_theta=5e-3, best_core=1e-3)

    phase = state["phases"]["m2b"]
    assert phase["gate"]["pass"] is False
    assert phase["fixed_slot"]["status"] == "skipped"
    assert phase["fixed_slot"]["reason"] == "mmlu_gate_failed"
    assert phase["full_eval"] == {}


def test_run_m2b_dry_run_skips_real_eval_merges(tmp_path, monkeypatch):
    args = make_args(tmp_path, dry_run=True)
    state: dict[str, object] = {}

    def fake_start_training_task(spec, gpu_id, workdir, python_bin, hf_endpoint, dry_run):
        output_dir = workdir / spec.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return rollout.RunningTask(spec.name, output_dir, gpu_id, process=None)

    def fake_evaluate_adapter(output_dir, gpu_id, workdir, python_bin, hf_endpoint, benchmarks, output_name, **kwargs):
        return output_dir / output_name

    def fail_start_subprocess(command, env, dry_run):
        raise AssertionError("Dry-run should not launch benchmark subprocesses.")

    def fail_merge(mmlu_path, partial_path, merged_path):
        raise AssertionError("Dry-run should not merge benchmark files.")

    monkeypatch.setattr(rollout, "start_training_task", fake_start_training_task)
    monkeypatch.setattr(rollout, "evaluate_adapter", fake_evaluate_adapter)
    monkeypatch.setattr(rollout, "start_subprocess", fail_start_subprocess)
    monkeypatch.setattr(rollout, "merge_full_benchmarks", fail_merge)

    rollout.run_m2b(args, state, best_theta=5e-3, best_core=1e-3)

    phase = state["phases"]["m2b"]
    assert phase["gate"]["pass"] is True
    assert phase["fixed_slot"]["status"] == "completed"
    assert set(phase["full_eval"]) == {"m2b_jora", "m2b_lora_r1", "m2b_lora_r2", "m2b_fixed_jora"}
    assert set(phase["full_eval"].values()) == {"dry-run"}
