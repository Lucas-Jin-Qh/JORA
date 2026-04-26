#!/usr/bin/env python3
"""
JORA experiment launcher.

Strips `_`-prefixed metadata fields from JSON configs before passing to train.py.
Also handles HF offline mode and GPU assignment.

Usage:
    python scripts/run_jora_exp.py configs/run_diag_main.json --gpu 0 --max_steps 5
    python scripts/run_jora_exp.py configs/run_diag_main.json --gpu 0  # full run
"""
import argparse
import json
import os
import subprocess
import sys


def load_json_stripped(path):
    """Load JSON, stripping `_`-prefixed metadata fields."""
    with open(path) as f:
        data = json.load(f)
    # Strip metadata fields (start with _)
    stripped = {k: v for k, v in data.items() if not k.startswith("_")}
    return stripped


def write_temp_json(data, output_dir):
    """Write stripped JSON to a temp file."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json", prefix="jora_exp_")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run JORA experiment from JSON config")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--num_train_epochs", type=float, default=None, help="Override num_train_epochs")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--dry_run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    # Load and strip config
    config = load_json_stripped(args.config)

    # Override fields
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.num_train_epochs is not None:
        config["num_train_epochs"] = args.num_train_epochs
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    # Always disable wandb / third-party loggers in offline mode
    config.setdefault("report_to", "none")

    # Write temp stripped JSON
    temp_path = write_temp_json(config, config.get("output_dir", "/tmp"))

    # Build command
    train_py = os.path.join(os.path.dirname(__file__), "..", "examples", "sft", "train.py")
    train_py = os.path.normpath(train_py)

    cmd = [
        sys.executable,
        train_py,
        temp_path,
    ]

    # Environment: GPU, offline HF
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"

    cmd_str = " ".join(cmd)
    print(f"[run_jora_exp] GPU={args.gpu}, config={args.config}")
    print(f"[run_jora_exp] output_dir={config.get('output_dir', 'N/A')}")
    print(f"[run_jora_exp] max_steps={config.get('max_steps', 'N/A (epochs)')}")
    print(f"[run_jora_exp] seed={config.get('seed', 'N/A')}")
    print(f"[run_jora_exp] cmd: {cmd_str}")

    if args.dry_run:
        print("[run_jora_exp] Dry run - not executing")
        os.unlink(temp_path)
        return

    result = subprocess.run(cmd, env=env)
    os.unlink(temp_path)

    if result.returncode != 0:
        print(f"[run_jora_exp] FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"[run_jora_exp] DONE")


if __name__ == "__main__":
    main()
