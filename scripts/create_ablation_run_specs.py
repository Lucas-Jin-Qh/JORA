#!/usr/bin/env python3
"""Create run_specs for remaining ablation experiments (seed=42 only).

Each ablation is a delta from the jora_s96_k16_seed42 anchor.
Writes run_spec.json and run_command.sh for each.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHOR_SPEC_PATH = REPO_ROOT / "formal_runs/three_gpu_bf16_phase3_global_queue/anchors/jora_s96_k16_seed42/run_spec.json"
ABL_DIR = REPO_ROOT / "formal_runs/ablation_seed42"
PYTHON_BIN = "/home/jqh/miniconda3/envs/peft-jora/bin/python"
TRAIN_SCRIPT = str(REPO_ROOT / "examples/sft/train.py")

PYTHONPATH = "/home/jqh/Workshop/JORA/src:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
HF_HOME = "/home/jqh/Workshop/JORA/formal_runs/three_gpu_bf16/hf_home"
HF_DATASETS_CACHE = "/home/jqh/Workshop/JORA/formal_runs/three_gpu_bf16/hf_datasets"


def load_anchor() -> dict:
    return json.loads(ANCHOR_SPEC_PATH.read_text())


def base_command(anchor: dict, output_dir: str) -> list[str]:
    """Reconstruct the anchor command, replacing output_dir."""
    cmd = list(anchor["command"])
    idx = cmd.index("--output_dir")
    cmd[idx + 1] = output_dir
    return cmd


def patch_command(cmd: list[str], **kwargs) -> list[str]:
    """Patch or insert CLI args. kwargs maps flag_name -> value.
    flag_name uses underscores, will be emitted as --flag_name.
    """
    cmd = list(cmd)
    for flag, value in kwargs.items():
        flag_str = f"--{flag}"
        if flag_str in cmd:
            idx = cmd.index(flag_str)
            cmd[idx + 1] = str(value)
        else:
            cmd.extend([flag_str, str(value)])
    return cmd


def write_ablation(
    name: str,
    gpu: int,
    anchor: dict,
    changes: dict,
    cmd_extra: dict,
) -> None:
    output_dir = str(ABL_DIR / name)
    output_dir_path = Path(output_dir)

    if output_dir_path.exists():
        existing_files = list(output_dir_path.iterdir())
        non_spec = [f for f in existing_files if f.name not in ("run_spec.json", "run_command.sh")]
        if non_spec:
            print(f"  SKIP {name}: directory has training artifacts, not overwriting")
            return

    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = base_command(anchor, output_dir)
    cmd = patch_command(cmd, **cmd_extra)

    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": PYTHONPATH,
        "HF_HOME": HF_HOME,
        "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
        "HF_HUB_DISABLE_XET": "1",
    }

    run_spec = {
        "name": name,
        "baseline": str(ANCHOR_SPEC_PATH),
        "changes": {"seed": 42, "gpu": gpu, "output_dir": output_dir, **changes},
    }

    (output_dir_path / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )

    sh_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        f"export PYTHONPATH={shlex.quote(PYTHONPATH)}",
        f"export HF_HOME={shlex.quote(HF_HOME)}",
        f"export HF_DATASETS_CACHE={shlex.quote(HF_DATASETS_CACHE)}",
        "export HF_HUB_DISABLE_XET=1",
        "export TOKENIZERS_PARALLELISM=false",
        " ".join(shlex.quote(c) for c in cmd),
        "",
    ]
    (output_dir_path / "run_command.sh").write_text("\n".join(sh_lines), encoding="utf-8")
    (output_dir_path / "run_command.sh").chmod(0o755)

    print(f"  OK {name} -> {output_dir} (GPU {gpu})")
    print(f"     changes: {changes}")


def main() -> None:
    anchor = load_anchor()
    print(f"Anchor: {ANCHOR_SPEC_PATH}")

    ablations = [
        # (name, gpu, changes dict, cmd_extra dict)
        (
            "abl_oer",
            0,
            {"jora_magnitude": "oer_softmax"},
            {"jora_magnitude": "oer_softmax"},
        ),
        (
            "abl_k32",
            2,
            {"jora_k": 32},
            {"jora_k": 32},
        ),
        (
            "abl_warmup500",
            0,
            {"jora_t_stat": 500},
            {"jora_t_stat": 500},
        ),
        (
            "abl_highlow",
            2,
            {"jora_pairing_strategy": "high_low"},
            {"jora_pairing_strategy": "high_low"},
        ),
        (
            "abl_lr_theta_003",
            0,
            {"jora_lr_theta": 0.003},
            {"jora_lr_theta": 0.003},
        ),
        (
            "abl_lr_theta_005",
            2,
            {"jora_lr_theta": 0.005},
            {"jora_lr_theta": 0.005},
        ),
    ]

    for name, gpu, changes, cmd_extra in ablations:
        write_ablation(name, gpu, anchor, changes, cmd_extra)

    print("\nDone.")


if __name__ == "__main__":
    main()
