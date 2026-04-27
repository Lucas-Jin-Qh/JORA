# JORA Repository Map

Last updated: 2026-04-26

This document is a static map of the current JORA repository. It is intended to help research and engineering work stay aligned with the current code, configs, experiments, and known risks.

## 1. JORA implementation files

### Core package
- `src/peft/tuners/jora/__init__.py` — package exports.
- `src/peft/tuners/jora/config.py` — `JoraConfig` definition, factory methods such as `paper_path()`, `diag_path()`, and `selective_path()`.
- `src/peft/tuners/jora/model.py` — PEFT model wrapper integration.
- `src/peft/tuners/jora/layer.py` — main `JoraLayer` implementation, forward path, `compute_delta()`, merge/unmerge logic.
- `src/peft/tuners/jora/core.py` — core operator implementations including `SelectiveDiagCore`, `DiagCore`, `BlockCore`, `LowRankCore`, and core builder logic.
- `src/peft/tuners/jora/rotation.py` — sparse Givens rotation utilities.
- `src/peft/tuners/jora/selection.py` — pair/slot selection logic and calibration-related utilities.
- `src/peft/tuners/jora/magnitude.py` — magnitude scaling such as OER/ECD paths.
- `src/peft/tuners/jora/callbacks.py` — trainer callbacks for JORA-specific update scheduling and metrics.
- `src/peft/tuners/jora/utils.py` — JORA utility helpers.
- `src/peft/tuners/jora/README.md` — package-level usage and parameter notes.

### High-value implementation notes
- `src/peft/tuners/jora/layer.py:488` — `SelectiveDiagCore` uses the paper-exact residualized path.
- `src/peft/tuners/jora/layer.py:543` — `DiagCore` / `BlockCore` / `LowRankCore` currently use the legacy additive path.
- `src/peft/tuners/jora/layer.py:867` — `SelectiveDiagCore` merge path is handled separately from legacy core types.
- `src/peft/tuners/jora/config.py:143` — `paper_path()` remains selective/paper-path oriented.
- `src/peft/tuners/jora/config.py:176` — `diag_path()` factory for JORA-Diag.
- `src/peft/tuners/jora/config.py:219` — `selective_path()` factory for JORA-Selective.

## 2. Training entrypoints

### Primary experiment launcher
- `scripts/run_jora_exp.py:37` — main JSON-driven launcher for JORA experiments.
  - Loads JSON config.
  - Strips `_`-prefixed metadata fields.
  - Applies overrides such as `--gpu`, `--max_steps`, `--num_train_epochs`, `--output_dir`, and `--seed`.
  - Launches `examples/sft/train.py`.

### Main training implementation
- `examples/sft/train.py:527` — main supervised fine-tuning entrypoint.
- `examples/sft/train.py:595` — `HfArgumentParser` entrypoint for JSON/CLI configs.
- `examples/sft/train.py:564` — installs `JoraTrainerCallback` and `JoraMetricsCallback` when using JORA.
- `examples/sft/train.py` depends on local `utils.create_and_prepare_model()` and `create_datasets()` to construct the model and datasets.

### Supporting training utilities
- `train_with_config.py:421` — command generator / estimator for PEFT training configs; useful for planning and parameter-count estimation, not the main execution path.
- `smoke_test_opt350m.py:21` — small local smoke test for JORA/OPT-350M wiring.
- `scripts/jora_cli_smoke.py:201` — CLI-oriented smoke test utility.
- `scripts/run_selective_diag_5run.py:226` — selective-diag specific rollout script.
- `scripts/jora_sweep.py:339` — experiment sweep helper.
- `scripts/run_phase2_three_seed_rollout.py:852` — phased rollout orchestrator for three-seed experiments.
- `scripts/run_phase3_full_epoch_rollout.py:932` — phased full-epoch rollout orchestrator.
- `scripts/single_gpu_bf16_plan.py:713` — single-GPU BF16 planning/orchestration helper.

## 3. Eval entrypoints

### Generic PEFT evaluation
- `evaluate_peft_model.py:43` — evaluates a PEFT checkpoint with `lm_eval.simple_evaluate`.
- `evaluate_peft_model.py:281` — CLI entrypoint.

### Custom PEFT evaluation
- `evaluate_peft_model_custom.py:18` — customized PEFT evaluation path using `lm_eval` and HF models.
- `evaluate_peft_model_custom.py:282` — CLI entrypoint.

### Reasoning benchmark evaluation
- `scripts/evaluate_reasoning_benchmarks.py:315` — benchmark runner for reasoning tasks.
  - Includes task-oriented evaluation utilities for MMLU/ARC-style multiple-choice and related reporting.

### Analysis / diagnostics helpers
- `scripts/paired_error_analysis.py:314` — paired comparison analysis helper.
- `verify_config.py:34` — quick configuration/result sanity script with hard-coded comparison notes.
- `diagnose_jora.py` — JORA-specific diagnostic script.
- `diagnose_gradient.py` — gradient-related diagnosis helper.

## 4. Config files for JORA-Diag, JORA-NoRot, JORA-Selective

### JORA-Diag configs
- `configs/run_diag_main.json` — main JORA-Diag config.
- `configs/run_diag_main_s42.json` — seed-42 JORA-Diag run.
- `configs/run_diag_main_s7.json` — seed-7 JORA-Diag run.
- `configs/run_diag_main_s2023.json` — seed-2023 JORA-Diag run.
- `configs/run_diag_main_s42_3ep.json` — 3-epoch JORA-Diag run.
- `configs/run_init_ablation_zero.json` — init ablation variant tied to JORA-Diag framing.
- `configs/run_init_ablation_nonzero_zero_core.json` — init ablation variant.
- `configs/run_init_ablation_zero_both.json` — init ablation variant.
- `configs/run_init_ablation_nonzero_zero_core_3ep.json` — longer init-ablation run.
- `configs/run_lr_theta_1x_s42.json` — theta-LR ablation.
- `configs/run_lr_theta_5x_s42.json` — theta-LR ablation.
- `configs/run_lr_theta_10x_s42.json` — theta-LR ablation.
- `configs/run_rotation_only_s42.json` — rotation-only variant related to the JORA-Diag mechanism story.
- `configs/run_jora_block.json` — block-core family variant.
- `configs/run_jora_lowrank.json` — low-rank family variant.

### JORA-NoRot configs
- `configs/run_diag_no_rotation.json` — main no-rotation baseline.
- `configs/run_diag_no_rotation_s42.json` — seed-42 no-rotation run.
- `configs/run_diag_no_rotation_s7.json` — seed-7 no-rotation run.
- `configs/run_diag_no_rotation_s2023.json` — seed-2023 no-rotation run.
- `configs/run_diag_no_rotation_s42_3ep.json` — 3-epoch no-rotation run.
- `configs/run_diag_only_s42.json` — closely related diagonal-only naming variant.

### JORA-Selective configs
- `configs/run1_selective_diag_baseline.json` — selective-diag baseline.
- `configs/run2_selective_diag_paramgroup.json` — selective-diag parameter-group experiment.

### Experiment playbook
- `configs/run_experiments.md` — gate-based execution playbook and current experiment process baseline.

## 5. Existing result files

### Main result directories under `results/`
- `results/run_diag_main_s42`
- `results/run_diag_main_s42_3ep`
- `results/run_diag_no_rotation_s42`
- `results/run_diag_no_rotation_s42_3ep`
- `results/run_init_ablation_nonzero_zero_core`
- `results/run_init_ablation_nonzero_zero_core_3ep`
- `results/run_init_ablation_zero_both`
- `results/run_rotation_only_s42`

### Typical contents in each result directory
Most run directories contain a similar artifact set:
- `adapter_config.json`
- `adapter_model.safetensors`
- `training_args.bin`
- tokenizer artifacts such as `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`
- optional diagnostics such as `jora_diagnostics.csv`
- TensorBoard event files under `runs/`

### Logs
- `logs/run_diag_main_s42.log`
- `logs/run_diag_main_s42_3ep.log`
- `logs/run_diag_no_rotation_s42.log`
- `logs/run_diag_no_rotation_s42_3ep.log`
- `logs/run_init_ablation_nonzero_zero_core.log`
- `logs/run_init_ablation_nonzero_zero_core_3ep.log`
- `logs/run_init_ablation_zero_both.log`
- `logs/run_rotation_only_s42.log`
- `logs/phase0_baseline.log`
- `logs/PHASES_0_6_EXECUTION_LOG.md`

### Formal run artifacts
- `formal_runs/selective_diag_5run/run_results.json` — aggregated selective-diag run results.
- `formal_runs/selective_diag_5run/phase_a/...` — archived selective-diag phase outputs.

### Other top-level result/summary files
- `baseline_results.json` — top-level baseline summary artifact.

## 6. Known risks and TODOs

### Hard risks already encoded in `AGENTS.md`
- `AGENTS.md:5` — current main method is JORA-Diag.
- `AGENTS.md:6` — current risk: rotation contribution is unproven; JORA-NoRot is a first-class baseline.
- `AGENTS.md:9` — do not claim rotation drives gains unless matched JORA-Diag vs JORA-NoRot evidence supports it.
- `AGENTS.md:10` — do not write final Method formula until `FORMULA_AUDIT.md` is complete.
- `AGENTS.md:11` — do not run expensive sweeps before sanity/save-load/merge checks pass.

### Method / implementation risks
- `src/peft/tuners/jora/layer.py:488` vs `src/peft/tuners/jora/layer.py:543` — the repository currently contains two different operator semantics:
  - `SelectiveDiagCore`: paper-exact residualized path.
  - `DiagCore` / `BlockCore` / `LowRankCore`: legacy additive path.
- This means the final paper Method section must not assume a unified formula without a formula audit.
- Rotation contribution is currently not established by the known 1-epoch evidence captured in `AGENTS.md`.
- Merge behavior is not uniform across all JORA variants; `SelectiveDiagCore` has a dedicated path while other cores use a more conservative legacy approximation.

### Experiment / evidence risks
- Current evidence recorded in `AGENTS.md`:
  - JORA-Diag 1 epoch train loss: `2.23870`
  - JORA-NoRot 1 epoch train loss: `2.23866`
  - interpretation: no discernible 1-epoch train-loss gain from rotation, with a much higher runtime cost for rotation-on.
- This makes JORA-NoRot a claim-determining baseline, not a cosmetic ablation.
- Longer-horizon matched ON/OFF evidence remains important for any stronger statement about rotation.

### Testing and verification assets
- `tests/test_jora.py` — broad JORA tests.
- `tests/test_jora_paper_path.py` — selective/paper-path focused tests.
- `tests/test_jora_diag_path.py` — diag-path focused tests.
- Additional rollout/orchestration tests:
  - `tests/test_run_phase2_three_seed_rollout.py`
  - `tests/test_run_phase3_full_epoch_rollout.py`

### TODO sources worth checking before future work
- `TODO.md` — general project TODO list.
- `configs/run_experiments.md` — current reliability-first execution plan.
- `JORA_Complete_Technical_Reference.md` — large historical/theoretical reference; useful context but may contain claims that are stronger than the latest empirical position.
- `JORA_MERGE_README.md` — merge-related notes.
- `JORA_Technical_Update_0329.md` — update notes and potentially stale assumptions.

## Suggested reading order for a new contributor
1. `AGENTS.md`
2. `configs/run_experiments.md`
3. `src/peft/tuners/jora/config.py`
4. `src/peft/tuners/jora/core.py`
5. `src/peft/tuners/jora/layer.py`
6. `tests/test_jora_diag_path.py`
7. `tests/test_jora_paper_path.py`
8. `scripts/run_jora_exp.py`
9. `examples/sft/train.py`
10. `logs/run_diag_main_s42.log` and `logs/run_diag_no_rotation_s42.log`

## Current bottom line
- The repository is organized around `JORA-Diag` as the practical main line.
- `JORA-NoRot` is already a first-class baseline in both method logic and experiment interpretation.
- `SelectiveDiagCore` remains important because it is the cleanest paper-exact path in the current codebase.
- The biggest research risk is still unresolved: whether rotation contributes anything beyond diagonal adaptation.
