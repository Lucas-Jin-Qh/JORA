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


phase3 = _load_script_module("run_phase3_full_epoch_rollout", "scripts/run_phase3_full_epoch_rollout.py")


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
        gpu_free_memory_mb=16_000,
        gpu_free_utilization=100,
        core_sweep_epochs=1.0,
        core_sweep_mmlu_limit=200,
        shape_probe_epochs=1.0,
        shape_probe_mmlu_limit=200,
        anchor_epochs=3.0,
        appendix_core_epochs=3.0,
        jora_lr_theta=None,
        jora_lr_core=None,
        dry_run=dry_run,
    )


def test_locked_candidates_and_default_schedule():
    assert phase3.CLAIM_SEEDS == (42, 1337, 2026)
    assert [candidate.tag for candidate in phase3.CORE_CANDIDATES] == [
        "selective_diag_s32_k32",
        "diag",
        "block_bs4",
        "lowrank_r1",
    ]
    assert [candidate.tag for candidate in phase3.APPENDIX_CORE_CANDIDATES] == ["diag", "block_bs4", "lowrank_r1"]
    assert [candidate.tag for candidate in phase3.SHAPE_CANDIDATES] == ["s32_k32", "s96_k16", "s96_k32"]


def test_build_core_sweep_specs_use_epoch_schedule():
    specs = phase3.build_core_sweep_specs(best_theta=1e-3, best_core=5e-4, num_train_epochs=1.0)
    spec_map = {spec.output_subdir: spec for spec in specs}

    assert len(specs) == 6
    assert spec_map["core_sweep/jora_lowrank_r1_seed42"].num_train_epochs == 1.0
    assert spec_map["core_sweep/jora_lowrank_r1_seed42"].max_steps is None
    assert spec_map["core_sweep/jora_lowrank_r1_seed42"].save_strategy == "epoch"
    assert spec_map["core_sweep/jora_block_bs4_seed42"].jora_block_size == 4
    assert spec_map["core_sweep/lora_r2_seed42"].method == "lora"


def test_build_anchor_and_appendix_specs_are_full_epoch():
    selected_shape = phase3.ShapeCandidate(tag="s96_k16", s_l=96, s_r=96, k=16)

    anchor_specs = phase3.build_anchor_specs(selected_shape, best_theta=1e-3, best_core=5e-4, seed=1337, num_train_epochs=3.0)
    all_anchor_specs = phase3.build_all_anchor_specs(selected_shape, best_theta=1e-3, best_core=5e-4, num_train_epochs=3.0)
    appendix_specs = phase3.build_appendix_core_specs(best_theta=1e-3, best_core=5e-4, num_train_epochs=3.0)

    assert [spec.name for spec in anchor_specs] == [
        "anchor_jora_s96_k16_seed1337",
        "anchor_lora_r1_seed1337",
        "anchor_lora_r2_seed1337",
    ]
    assert [spec.name for spec in all_anchor_specs[:3]] == [
        "anchor_jora_s96_k16_seed42",
        "anchor_jora_s96_k16_seed1337",
        "anchor_jora_s96_k16_seed2026",
    ]
    assert all(spec.num_train_epochs == 3.0 for spec in anchor_specs)
    assert all(spec.save_strategy == "epoch" for spec in anchor_specs)

    appendix_map = {spec.output_subdir: spec for spec in appendix_specs}
    assert set(appendix_map) == {
        "appendix_cores/jora_diag_seed42",
        "appendix_cores/jora_block_bs4_seed42",
        "appendix_cores/jora_lowrank_r1_seed42",
    }
    assert appendix_map["appendix_cores/jora_lowrank_r1_seed42"].jora_lowrank_r == 1


def test_entry_from_result_reads_parameter_summary_field(tmp_path):
    spec = phase3.build_anchor_specs(
        phase3.ShapeCandidate(tag="s32_k32", s_l=32, s_r=32, k=32),
        best_theta=1e-3,
        best_core=5e-4,
        seed=42,
        num_train_epochs=3.0,
    )[0]
    output_dir = tmp_path / "anchors" / "jora_s32_k32_seed42"
    output_dir.mkdir(parents=True)
    result_path = output_dir / phase3.FULL_MMLU_OUTPUT_NAME
    result_path.write_text(
        json.dumps(
            {
                "parameters": {"trainable_params": 8192},
                "benchmarks": {"mmlu": {"accuracy": 0.55}},
            }
        ),
        encoding="utf-8",
    )

    entry = phase3.entry_from_result(spec, output_dir, result_path, gpu_id=0, dry_run=False)

    assert entry["trainable_params"] == 8192
    assert entry["mmlu_accuracy"] == 0.55


def test_write_phase3_summary_includes_appendix_core_block(tmp_path):
    args = make_args(tmp_path)
    state: dict[str, object] = {}
    core_payload = {"leaderboard": [{"family": "jora_s32_k32", "mmlu_accuracy": 0.4}]}
    shape_payload = {
        "leaderboard": [{"family": "jora_s32_k32", "mmlu_accuracy": 0.41}],
        "selected": {"tag": "s32_k32", "s_l": 32, "s_r": 32, "k": 32},
    }
    anchor_payload = {"selected_shape": {"tag": "s32_k32", "s_l": 32, "s_r": 32, "k": 32}, "rounds": {}}

    bench_path = tmp_path / "appendix_block.json"
    bench_path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    "mmlu": {"accuracy": 0.42},
                    "arc_challenge": {"accuracy": 0.5},
                    "gsm8k": {"accuracy": 0.1},
                },
                "average_accuracy": 0.34,
            }
        ),
        encoding="utf-8",
    )
    appendix_payload = {
        "runs": {
            "appendix_jora_block_bs4_seed42": {
                "family": "jora_block_bs4",
                "method": "jora",
                "seed": 42,
                "output_dir": str(tmp_path / "appendix_cores" / "jora_block_bs4_seed42"),
                "trainable_params": 1052672,
                "benchmarks_output": str(bench_path),
            }
        }
    }

    summary = phase3.write_phase3_summary(args, state, core_payload, shape_payload, anchor_payload, appendix_payload)

    assert summary["appendix_cores"]["families"]["jora_block_bs4"]["benchmarks"]["mmlu"]["mean"] == 0.42
    assert (tmp_path / "summary.json").exists()
