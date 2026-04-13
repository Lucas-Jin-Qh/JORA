from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
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


plan = _load_script_module("single_gpu_bf16_plan", "scripts/single_gpu_bf16_plan.py")


def make_override_args(**overrides):
    defaults = {
        "seed": None,
        "model_profile": None,
        "model_name_or_path": None,
        "target_modules": None,
        "dataset_name": None,
        "max_steps": None,
        "num_train_epochs": None,
        "learning_rate": None,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "max_length": None,
        "jora_lr_theta": None,
        "jora_lr_core": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_jora_anchor_command_uses_claim_scope_and_bf16():
    spec = plan._base_named_specs()["m2b_jora"]
    command = plan.build_train_command(spec, Path("/tmp/out"), sys.executable)
    joined = " ".join(command)
    assert "--use_peft_jora True" in joined
    assert "--jora_target_modules q_proj,o_proj" in joined
    assert "--jora_core selective_diag" in joined
    assert "--jora_k 32" in joined
    assert "--torch_dtype bfloat16" in joined
    assert "--bf16 True" in joined
    assert "--use_4bit_quantization" not in joined


def test_lora_anchor_command_uses_matching_scope():
    spec = plan._base_named_specs()["m2b_lora_r2"]
    command = plan.build_train_command(spec, Path("/tmp/out"), sys.executable)
    joined = " ".join(command)
    assert "--use_peft_lora True" in joined
    assert "--lora_target_modules q_proj,o_proj" in joined
    assert "--lora_r 2" in joined
    assert "--lora_alpha 4" in joined


def test_non_selective_jora_command_emits_core_specific_flags():
    base_spec = plan._base_named_specs()["m2b_jora"]
    spec = replace(
        base_spec,
        jora_core="lowrank",
        jora_lowrank_r=1,
        jora_lowrank_alpha=1.0,
        jora_zero_init_core=True,
    )
    command = plan.build_train_command(spec, Path("/tmp/out"), sys.executable)
    joined = " ".join(command)

    assert "--jora_core lowrank" in joined
    assert "--jora_lowrank_r 1" in joined
    assert "--jora_lowrank_alpha 1" in joined
    assert "--jora_zero_init_core True" in joined


def test_m1_sweep_has_expected_grid_size():
    sweep_specs = plan._m1_sweep_specs()
    assert len(sweep_specs) == 12
    assert {spec.jora_lr_theta for spec in sweep_specs} == {1e-3, 5e-3, 1e-2, 5e-2}
    assert {spec.jora_lr_core for spec in sweep_specs} == {5e-4, 1e-3, 5e-3}


def test_build_env_uses_shared_hf_caches_across_workdirs(tmp_path):
    env_a = plan.build_env(tmp_path / "run_a", None)
    env_b = plan.build_env(tmp_path / "run_b", None)

    assert env_a["HF_HOME"] == env_b["HF_HOME"] == str(plan.shared_hf_home_path())
    assert env_a["HF_DATASETS_CACHE"] == env_b["HF_DATASETS_CACHE"] == str(plan.shared_hf_datasets_cache_path())
    assert env_a["HF_HUB_DISABLE_XET"] == env_b["HF_HUB_DISABLE_XET"] == "1"
    assert env_a["HF_DATASETS_OFFLINE"] == env_b["HF_DATASETS_OFFLINE"] == "1"
    assert env_a["HF_HUB_OFFLINE"] == env_b["HF_HUB_OFFLINE"] == "1"
    assert env_a["TRANSFORMERS_OFFLINE"] == env_b["TRANSFORMERS_OFFLINE"] == "1"
    assert env_a["MPLCONFIGDIR"] == str(tmp_path / "run_a" / "mplconfig")
    assert env_b["MPLCONFIGDIR"] == str(tmp_path / "run_b" / "mplconfig")


def test_mistral_specs_can_use_local_checkpoint(monkeypatch):
    monkeypatch.setattr(plan, "default_mistral_model_name_or_path", lambda: "/mnt/local/Mistral-7B-v0.1")
    specs = plan._base_named_specs()

    assert specs["m2a_jora"].model_name_or_path == "/mnt/local/Mistral-7B-v0.1"
    assert specs["m2b_jora"].model_name_or_path == "/mnt/local/Mistral-7B-v0.1"
    assert specs["m2b_lora_r2"].model_name_or_path == "/mnt/local/Mistral-7B-v0.1"


def test_gpt2_profile_override_updates_model_and_targets(monkeypatch):
    monkeypatch.setattr(plan, "default_gpt2_large_model_name_or_path", lambda: "/mnt/local/gpt2-large")
    spec = plan._base_named_specs()["m2a_jora"]

    overridden = plan._apply_overrides(spec, make_override_args(model_profile="gpt2-large"))

    assert overridden.model_profile == "gpt2-large"
    assert overridden.model_name_or_path == "/mnt/local/gpt2-large"
    assert overridden.target_modules == "attn.c_proj"


def test_gpt2_profile_command_uses_attention_suffix_targets(monkeypatch):
    monkeypatch.setattr(plan, "default_gpt2_large_model_name_or_path", lambda: "/mnt/local/gpt2-large")
    spec = plan._apply_overrides(plan._base_named_specs()["m2a_jora"], make_override_args(model_profile="gpt2-large"))
    command = plan.build_train_command(spec, Path("/tmp/out"), sys.executable)
    joined = " ".join(command)

    assert "--model_name_or_path /mnt/local/gpt2-large" in joined
    assert "--jora_target_modules attn.c_proj" in joined


def test_prepare_output_dir_for_launch_clears_manifest_only_retry_dir(tmp_path):
    output_dir = tmp_path / "retryable"
    output_dir.mkdir()
    (output_dir / "run_spec.json").write_text("{}", encoding="utf-8")
    (output_dir / "run_command.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    plan.prepare_output_dir_for_launch(output_dir)

    assert not output_dir.exists()


def test_epoch_based_spec_emits_num_train_epochs_not_max_steps():
    spec = replace(
        plan._base_named_specs()["m2b_jora"],
        max_steps=None,
        num_train_epochs=3.0,
        save_strategy="epoch",
        save_total_limit=2,
    )
    command = plan.build_train_command(spec, Path("/tmp/out"), sys.executable)
    joined = " ".join(command)

    assert "--num_train_epochs 3" in joined
    assert "--max_steps" not in joined
    assert "--save_strategy epoch" in joined
    assert "--save_total_limit 2" in joined


def test_write_manifest_records_profile_target_modules_and_offline_env(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "default_gpt2_xl_model_name_or_path", lambda: "/mnt/local/gpt2-xl")
    spec = plan._apply_overrides(plan._base_named_specs()["m2a_lora_r1"], make_override_args(model_profile="gpt2-xl"))
    env = plan.build_env(tmp_path / "workdir", "https://hf-mirror.com")
    output_dir = tmp_path / "manifest_case"
    command = plan.build_train_command(spec, output_dir, sys.executable)

    plan.write_manifest(output_dir, spec, command, env)

    manifest = json.loads((output_dir / "run_spec.json").read_text(encoding="utf-8"))
    run_command = (output_dir / "run_command.sh").read_text(encoding="utf-8")

    assert manifest["spec"]["model_profile"] == "gpt2-xl"
    assert manifest["spec"]["target_modules"] == "attn.c_proj"
    assert manifest["claim_scope"]["model_profile"] == "gpt2-xl"
    assert manifest["claim_scope"]["target_modules"] == "attn.c_proj"
    assert manifest["environment"]["HF_DATASETS_OFFLINE"] == "1"
    assert manifest["environment"]["HF_HUB_OFFLINE"] == "1"
    assert manifest["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert manifest["environment"]["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert "export HF_DATASETS_OFFLINE=1" in run_command
    assert "export HF_HUB_OFFLINE=1" in run_command
    assert "export TRANSFORMERS_OFFLINE=1" in run_command
