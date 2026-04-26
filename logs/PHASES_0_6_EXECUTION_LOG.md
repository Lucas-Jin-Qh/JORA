# JORA Refactoring: Phase 0-6 Execution Log
# Date: 2026-04-24
# Executor: Claude Agent

## Summary

All phases 0-5 completed successfully. Code changes are verified and experiment configs are ready.

## Phase 0: Baseline Audit ✅

**Result**: All core unit tests pass. `paper_path()` defaults are correct. 
`core_init_std` is MISSING (expected - added in Phase 2).

**Key findings**:
- `JoraConfig()` default: `core=diag, theta_init_std=0.02, zero_init_core=False`
- `JoraConfig.paper_path()` default: `core=selective_diag, theta_init_std=0.0, zero_init_core=True`
- Known environment issue: HuggingFace Hub not accessible (use hf-mirror.com)

## Phase 1: Initialization Calibration ✅

**Result**: Phase 1 gate PASS. Safe init ranges confirmed.

**Recommendation**: `core_init_std = 5e-3`, `theta_init_std = 2e-3`

All cores (DiagCore, BlockCore, LowRankCore) pass the init bounds check:
- Zero-init: ||delta||/||x|| ≈ 0
- Nonzero init (1e-3 to 5e-3): ||delta||/||x|| < 1e-1

## Phase 2: Config Refactor ✅

**Changes** (`config.py`):
1. Added `core_init_std: float = 5e-3` field
2. Added `JoraConfig.diag_path()` factory method (main method)
3. Added `JoraConfig.selective_path()` factory method (efficiency variant)
4. `paper_path()` unchanged

**train.py changes**:
- Added `jora_theta_init_std` argument
- Added `jora_core_init_std` argument

**utils.py changes**:
- Added factory dispatch: `selective_diag → paper_path()`, `diag → diag_path()`
- Passed `theta_init_std` and `core_init_std` through kwargs

## Phase 3: Core Init Unification ✅

**Changes** (`core.py`):
1. `DiagCore.__init__`: added `init_std` parameter, uses `std * randn()` instead of hardcoded `0.01`
2. `BlockCore.__init__`: added `init_std` parameter, uses `init_std * randn()` instead of hardcoded `0.1`
3. `LowRankCore.__init__`: added `init_std` parameter, both A and B use `init_std * randn()` (zero_init: A=0, B=init_std)
4. `build_core()`: passes `init_std` from `cfg.core_init_std`

## Phase 4: Layer Init Configurable + Tanh Removed ✅

**Changes** (`layer.py`):
1. Simplified theta init: removed core-type override, fully config-driven
2. Removed tanh from DiagCore/BlockCore/LowRankCore path (these cores use zero_init_core=True)

## Phase 5: Smoke Tests + Experiment Configs ✅

**New test file**: `tests/test_jora_diag_path.py`
- Factory defaults for diag_path(), selective_path(), paper_path()
- Zero-init and nonzero-init delta ratios
- Theta init from config
- Param counts
- build_core with init_std
- Merge consistency
- Full forward/backward pass

**Experiment configs created**:
| Config | Method | Purpose |
|--------|--------|---------|
| `run_diag_main.json` | JORA-Diag | Main method |
| `run_diag_no_rotation.json` | Diag-only | Rotation ablation |
| `run_lora_baseline.json` | LoRA-r1 | Baseline comparison |
| `run_lora_r2_baseline.json` | LoRA-r2 | Baseline comparison |
| `run_jora_block.json` | JORA-Block | Family expansion |
| `run_jora_lowrank.json` | JORA-LowRank | Rotation mechanism bridge |
| `run_init_ablation_zero.json` | JORA-Diag-init-zero | Init ablation |

## Phase 6: Experiment Ladder

### Network Status
HuggingFace Hub is NOT accessible. Use `HF_ENDPOINT=https://hf-mirror.com` for all model downloads.

### Stage 6.1: Wiring & Stability (smoke)
```bash
# GPU 1: JORA-Diag
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python examples/sft/train.py \
    configs/run_diag_main.json --max_steps=5 --output_dir=/tmp/smoke_diag_main

# GPU 1: Diag-only  
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python examples/sft/train.py \
    configs/run_diag_no_rotation.json --max_steps=5 --output_dir=/tmp/smoke_norot

# GPU 2: LoRA-r1
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python examples/sft/train.py \
    configs/run_lora_baseline.json --max_steps=5 --output_dir=/tmp/smoke_lora_r1

# GPU 2: LoRA-r2
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python examples/sft/train.py \
    configs/run_lora_r2_baseline.json --max_steps=5 --output_dir=/tmp/smoke_lora_r2
```

**Gate**: No crash, loss is finite, no NaN/Inf.

### Stage 6.2: Main Comparison, Single Seed
After Stage 6.1 passes, run full 1-epoch training:
```bash
# All 4 methods: JORA-Diag, Diag-only, LoRA-r1, LoRA-r2
# Seeds: 42
```

### Stage 6.3: Main Comparison, 3 Seeds
After Stage 6.2 passes, run with seeds 42, 2023, 7.

### Stage 6.4: Initialization Ablation
```bash
# Run init_ablation configs with seed 42
```

### Stage 6.5: Family Expansion
```bash
# JORA-Block, JORA-LowRank, 1 seed each
```

### Stage 6.6: Extended Tasks/Models
```bash
# Mistral-7B or LLaMA-2-7B with GSM8K/MATH/HumanEval
```

## Stop Conditions

If any of these occur, STOP and reassess:
1. JORA-Diag 3-seed mean < Diag-only → rotation harmful, downgrade rotation claim
2. JORA-Diag 3-seed mean < LoRA-r1 - 2pp → method not competitive
3. Nonzero init causes instability → revert to zero init
4. JORA-LowRank >> JORA-Diag → novelty collapses to "LoRA + rotation"

## Files Modified

| File | Phase | Change |
|------|-------|--------|
| `src/peft/tuners/jora/config.py` | 2 | +core_init_std, +diag_path(), +selective_path() |
| `src/peft/tuners/jora/core.py` | 3 | init_std for DiagCore/BlockCore/LowRankCore |
| `src/peft/tuners/jora/layer.py` | 4 | theta init config-driven, -tanh |
| `examples/sft/train.py` | 2 | +jora_theta_init_std, +jora_core_init_std args |
| `examples/sft/utils.py` | 2 | factory dispatch, init params |
| `tests/test_jora_diag_path.py` | 5 | NEW - diag/selective path tests |
| `configs/run_diag_main.json` | 5 | NEW |
| `configs/run_diag_no_rotation.json` | 5 | NEW |
| `configs/run_lora_baseline.json` | 5 | NEW |
| `configs/run_lora_r2_baseline.json` | 5 | NEW |
| `configs/run_jora_block.json` | 5 | NEW |
| `configs/run_jora_lowrank.json` | 5 | NEW |
| `configs/run_init_ablation_zero.json` | 5 | NEW |
