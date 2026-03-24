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


phase2 = _load_script_module("run_phase2_three_seed_rollout", "scripts/run_phase2_three_seed_rollout.py")


def make_args(
    tmp_path: Path,
    *,
    phase1_workdir: Path | None = None,
    dry_run: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        workdir=tmp_path,
        phase1_workdir=phase1_workdir or (tmp_path / "phase1"),
        python_bin=sys.executable,
        hf_endpoint=None,
        gpus=[0, 1, 2],
        poll_interval=1,
        gpu_free_memory_mb=20_000,
        gpu_free_utilization=100,
        core_sweep_steps=500,
        core_sweep_mmlu_limit=200,
        shape_probe_steps=500,
        shape_probe_mmlu_limit=200,
        anchor_steps=2000,
        jora_lr_theta=None,
        jora_lr_core=None,
        dry_run=dry_run,
    )


def test_claim_seeds_and_shape_candidates_are_locked():
    assert phase2.CLAIM_SEEDS == (42, 1337, 2026)
    assert [candidate.tag for candidate in phase2.CORE_CANDIDATES] == [
        "selective_diag_s32_k32",
        "diag",
        "block_bs4",
        "lowrank_r1",
    ]
    assert [candidate.tag for candidate in phase2.SHAPE_CANDIDATES] == ["s32_k32", "s96_k16", "s96_k32"]
    assert phase2.shape_candidate_by_tag("s96_k16").k == 16


def test_eligible_gpu_ids_prioritize_free_memory(monkeypatch):
    monkeypatch.setattr(
        phase2,
        "query_gpu_states",
        lambda: {
            0: {"memory_total_mb": 32_607, "memory_used_mb": 10_000, "memory_free_mb": 22_607, "utilization": 12},
            1: {"memory_total_mb": 32_607, "memory_used_mb": 11_500, "memory_free_mb": 21_107, "utilization": 100},
            2: {"memory_total_mb": 32_607, "memory_used_mb": 18_000, "memory_free_mb": 14_607, "utilization": 5},
        },
    )

    assert phase2.eligible_gpu_ids([0, 1, 2], free_memory_mb=20_000, free_utilization=100, dry_run=False) == [0, 1]
    assert phase2.eligible_gpu_ids([0, 1, 2], free_memory_mb=20_000, free_utilization=50, dry_run=False) == [0]


def test_resolve_jora_lrs_reads_phase1_selected_lr(tmp_path):
    phase1_workdir = tmp_path / "phase1"
    selected_path = phase1_workdir / "m1" / "selected_lr.json"
    selected_path.parent.mkdir(parents=True)
    selected_path.write_text(
        json.dumps({"jora_lr_theta": 1e-3, "jora_lr_core": 5e-4}),
        encoding="utf-8",
    )
    args = make_args(tmp_path, phase1_workdir=phase1_workdir)

    theta, core, source = phase2.resolve_jora_lrs(args)

    assert theta == 1e-3
    assert core == 5e-4
    assert source["source"] == "phase1_selected_lr"
    assert source["selected_lr_path"] == str(selected_path)


def test_build_core_sweep_specs_include_all_jora_cores_and_lora_baselines():
    specs = phase2.build_core_sweep_specs(best_theta=1e-3, best_core=5e-4, max_steps=500)
    spec_map = {spec.output_subdir: spec for spec in specs}

    assert len(specs) == 6
    assert spec_map["core_sweep/jora_diag_seed42"].jora_core == "diag"
    assert spec_map["core_sweep/jora_diag_seed42"].jora_zero_init_core is True
    assert spec_map["core_sweep/jora_block_bs4_seed42"].jora_core == "block"
    assert spec_map["core_sweep/jora_block_bs4_seed42"].jora_block_size == 4
    assert spec_map["core_sweep/jora_lowrank_r1_seed42"].jora_core == "lowrank"
    assert spec_map["core_sweep/jora_lowrank_r1_seed42"].jora_lowrank_r == 1
    assert spec_map["core_sweep/lora_r2_seed42"].method == "lora"


def test_build_shape_probe_specs_include_s96_k16():
    specs = phase2.build_shape_probe_specs(best_theta=1e-3, best_core=5e-4, max_steps=500)
    spec_map = {spec.output_subdir: spec for spec in specs}

    assert len(specs) == 3
    assert "shape_probe/jora_s96_k16_seed42" in spec_map
    assert spec_map["shape_probe/jora_s96_k16_seed42"].jora_s_l == 96
    assert spec_map["shape_probe/jora_s96_k16_seed42"].jora_k == 16
    assert spec_map["shape_probe/jora_s32_k32_seed42"].seed == 42


def test_select_best_shape_prefers_accuracy_then_lower_params():
    leaderboard = [
        {"tag": "s96_k32", "mmlu_accuracy": 0.41, "trainable_params": 16384, "k": 32, "s_l": 96, "name": "c"},
        {"tag": "s96_k16", "mmlu_accuracy": 0.41, "trainable_params": 8192, "k": 16, "s_l": 96, "name": "b"},
        {"tag": "s32_k32", "mmlu_accuracy": 0.39, "trainable_params": 8192, "k": 32, "s_l": 32, "name": "a"},
    ]
    leaderboard.sort(
        key=lambda row: (
            -row["mmlu_accuracy"],
            float("inf") if row["trainable_params"] is None else row["trainable_params"],
            row["k"],
            row["s_l"],
            row["name"],
        )
    )

    selected = phase2.select_best_shape(leaderboard)

    assert selected["tag"] == "s96_k16"


def test_run_core_sweep_writes_leaderboard_state(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    state: dict[str, object] = {}

    def fake_wait_for_gpus(*_args, **_kwargs):
        return None

    def fake_run_parallel_specs_with_mmlu_eval(specs, gpu_ids, args, output_name, limit_mmlu=None):
        assert output_name == phase2.CORE_SWEEP_OUTPUT_NAME
        assert limit_mmlu == 200
        result = {}
        for index, spec in enumerate(specs):
            entry = {
                "output_dir": str(tmp_path / spec.output_subdir),
                "mmlu_output": str(tmp_path / spec.output_subdir / output_name),
                "gpu_id": index % 3,
                "seed": 42,
                "method": spec.method,
                "family": phase2.family_name_for_spec(spec),
                "mmlu_accuracy": 0.30 + 0.01 * index,
                "trainable_params": 8192 * (index + 1),
            }
            if spec.method == "jora":
                entry["jora_core"] = spec.jora_core
                entry["shape"] = {"tag": f"s{spec.jora_s_l}_k{spec.jora_k}", "s_l": spec.jora_s_l, "s_r": spec.jora_s_r, "k": spec.jora_k}
                if spec.jora_block_size is not None:
                    entry["jora_block_size"] = spec.jora_block_size
                if spec.jora_lowrank_r is not None:
                    entry["jora_lowrank_r"] = spec.jora_lowrank_r
                if spec.jora_lowrank_alpha is not None:
                    entry["jora_lowrank_alpha"] = spec.jora_lowrank_alpha
            if spec.method == "lora":
                entry["lora_r"] = spec.lora_r
            result[spec.name] = entry
        return result

    monkeypatch.setattr(phase2, "wait_for_gpus", fake_wait_for_gpus)
    monkeypatch.setattr(phase2, "run_parallel_specs_with_mmlu_eval", fake_run_parallel_specs_with_mmlu_eval)

    payload = phase2.run_core_sweep(args, state, best_theta=1e-3, best_core=5e-4)

    assert "leaderboard" in payload
    assert len(payload["leaderboard"]) == 6
    assert state["phases"]["core_sweep"]["leaderboard"][0]["mmlu_accuracy"] >= state["phases"]["core_sweep"]["leaderboard"][-1]["mmlu_accuracy"]
    assert any(row["family"] == "jora_lowrank_r1" for row in payload["leaderboard"])


def test_run_shape_probe_writes_selected_shape_state(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    state: dict[str, object] = {}

    def fake_wait_for_gpus(*_args, **_kwargs):
        return None

    def fake_run_parallel_specs_with_mmlu_eval(specs, gpu_ids, args, output_name, limit_mmlu=None):
        assert output_name == phase2.SHAPE_PROBE_OUTPUT_NAME
        assert limit_mmlu == 200
        return {
            specs[0].name: {
                "output_dir": str(tmp_path / specs[0].output_subdir),
                "mmlu_output": str(tmp_path / specs[0].output_subdir / output_name),
                "gpu_id": 0,
                "seed": 42,
                "method": "jora",
                "family": "jora_s32_k32",
                "mmlu_accuracy": 0.35,
                "trainable_params": 8192,
                "shape": {"tag": "s32_k32", "s_l": 32, "s_r": 32, "k": 32},
            },
            specs[1].name: {
                "output_dir": str(tmp_path / specs[1].output_subdir),
                "mmlu_output": str(tmp_path / specs[1].output_subdir / output_name),
                "gpu_id": 1,
                "seed": 42,
                "method": "jora",
                "family": "jora_s96_k16",
                "mmlu_accuracy": 0.37,
                "trainable_params": 8192,
                "shape": {"tag": "s96_k16", "s_l": 96, "s_r": 96, "k": 16},
            },
            specs[2].name: {
                "output_dir": str(tmp_path / specs[2].output_subdir),
                "mmlu_output": str(tmp_path / specs[2].output_subdir / output_name),
                "gpu_id": 2,
                "seed": 42,
                "method": "jora",
                "family": "jora_s96_k32",
                "mmlu_accuracy": 0.36,
                "trainable_params": 12288,
                "shape": {"tag": "s96_k32", "s_l": 96, "s_r": 96, "k": 32},
            },
        }

    monkeypatch.setattr(phase2, "wait_for_gpus", fake_wait_for_gpus)
    monkeypatch.setattr(phase2, "run_parallel_specs_with_mmlu_eval", fake_run_parallel_specs_with_mmlu_eval)

    selected = phase2.run_shape_probe(args, state, best_theta=1e-3, best_core=5e-4)

    assert selected.tag == "s96_k16"
    assert state["phases"]["shape_probe"]["selected"]["tag"] == "s96_k16"
    selected_path = tmp_path / "shape_probe" / "selected_shape.json"
    assert selected_path.exists()
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    assert payload["tag"] == "s96_k16"


def test_run_anchor_rounds_uses_locked_seeds_and_shape_naming(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    state: dict[str, object] = {}
    seen_spec_names: list[str] = []

    def fake_wait_for_gpus(*_args, **_kwargs):
        return None

    def fake_run_parallel_specs_with_mmlu_eval(specs, gpu_ids, args, output_name, limit_mmlu=None):
        result = {}
        for index, spec in enumerate(specs):
            seen_spec_names.append(spec.name)
            result[spec.name] = {
                "output_dir": str(tmp_path / spec.output_subdir),
                "mmlu_output": str(tmp_path / spec.output_subdir / output_name),
                "gpu_id": index,
                "seed": spec.seed,
                "method": spec.method,
                "family": phase2.family_name_for_spec(spec),
                "mmlu_accuracy": 0.4,
                "trainable_params": 8192 if spec.method == "jora" else 524288,
            }
        return result

    def fake_run_parallel_full_evals(runs, gpu_ids, args):
        return {spec_name: str(tmp_path / "benchmarks" / f"{spec_name}.json") for spec_name in runs}

    monkeypatch.setattr(phase2, "wait_for_gpus", fake_wait_for_gpus)
    monkeypatch.setattr(phase2, "run_parallel_specs_with_mmlu_eval", fake_run_parallel_specs_with_mmlu_eval)
    monkeypatch.setattr(phase2, "run_parallel_full_evals", fake_run_parallel_full_evals)

    selected_shape = phase2.ShapeCandidate(tag="s96_k16", s_l=96, s_r=96, k=16)
    payload = phase2.run_anchor_rounds(args, state, selected_shape, best_theta=1e-3, best_core=5e-4)

    assert set(payload["rounds"]) == {"seed42", "seed1337", "seed2026"}
    assert any("anchor_jora_s96_k16_seed1337" == name for name in seen_spec_names)
    assert state["phases"]["anchors"]["rounds"]["seed1337"]["runs"]["anchor_jora_s96_k16_seed1337"]["output_dir"].endswith(
        "anchors/jora_s96_k16_seed1337"
    )


def test_aggregate_anchor_results_summarizes_full_benchmarks(tmp_path):
    families = {
        "anchor_jora_s96_k16_seed42": ("jora_s96_k16", 42, 0.55, 0.70, 0.10, 8192),
        "anchor_jora_s96_k16_seed1337": ("jora_s96_k16", 1337, 0.57, 0.71, 0.12, 8192),
        "anchor_jora_s96_k16_seed2026": ("jora_s96_k16", 2026, 0.56, 0.69, 0.11, 8192),
    }
    rounds = {"seed42": {"seed": 42, "runs": {}}, "seed1337": {"seed": 1337, "runs": {}}, "seed2026": {"seed": 2026, "runs": {}}}
    for spec_name, (family, seed, mmlu, arc, gsm8k, params) in families.items():
        bench_path = tmp_path / f"{spec_name}.json"
        payload = {
            "benchmarks": {
                "mmlu": {"accuracy": mmlu},
                "arc_challenge": {"accuracy": arc},
                "gsm8k": {"accuracy": gsm8k},
            },
            "average_accuracy": (mmlu + arc + gsm8k) / 3.0,
        }
        bench_path.write_text(json.dumps(payload), encoding="utf-8")
        rounds[f"seed{seed}"]["runs"][spec_name] = {
            "family": family,
            "method": "jora",
            "seed": seed,
            "output_dir": str(tmp_path / spec_name),
            "trainable_params": params,
            "benchmarks_output": str(bench_path),
        }

    summary = phase2.aggregate_anchor_results(
        {
            "selected_shape": {"tag": "s96_k16", "s_l": 96, "s_r": 96, "k": 16},
            "rounds": rounds,
        }
    )

    family = summary["families"]["jora_s96_k16"]
    assert family["seeds"] == [42, 1337, 2026]
    assert family["trainable_params"]["mean"] == 8192
    assert round(family["benchmarks"]["mmlu"]["mean"], 4) == 0.56


def test_write_phase2_summary_includes_core_sweep(tmp_path):
    args = make_args(tmp_path)
    state: dict[str, object] = {}
    core_payload = {
        "leaderboard": [
            {"family": "jora_diag", "mmlu_accuracy": 0.32},
            {"family": "jora_s32_k32", "mmlu_accuracy": 0.31},
        ]
    }
    anchor_payload = {
        "selected_shape": {"tag": "s96_k16", "s_l": 96, "s_r": 96, "k": 16},
        "rounds": {},
    }

    summary = phase2.write_phase2_summary(args, state, anchor_payload, core_payload)

    assert "core_sweep" in summary
    assert summary["core_sweep"]["leaderboard"][0]["family"] == "jora_diag"
    assert (tmp_path / "summary.json").exists()
