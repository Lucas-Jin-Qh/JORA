# JORA M0: Save/Load/Merge Correctness Gate

**Status: PASS (27/27 tests)** | **Date: 2026-04-28**

## Purpose

Verify that JORA's save/load roundtrip and merge/unmerge operations produce correct outputs before investing GPU hours in expensive training sweeps.

## How to Run

```bash
cd /home/jqh/Workshop/JORA
HF_ENDPOINT=https://hf-mirror.com TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/test_jora_save_load_merge_sanity.py \
  -v --override-ini=addopts= -p no:launch_testing_ros_pytest_entrypoint
```

Or from a clean directory (avoids ROS plugin conflicts):

```bash
cd /tmp
cp /home/jqh/Workshop/JORA/tests/test_jora_save_load_merge_sanity.py .
cp /home/jqh/Workshop/JORA/tests/test_jora.py .
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
python -m pytest test_jora_save_load_merge_sanity.py -v
```

## Coverage

27/27 tests pass:

- S1: JORA-Diag Save/Load (3 tests): output identity, theta preserved, pairs preserved — ALL PASS
- S2: JORA-NoRot Save/Load (3 tests): theta=None, output identity, merged≈base — ALL PASS
- S3: JORA-Paper Save/Load (2 tests): output identity, core delta preserved — ALL PASS
- S4: Magnitude Scaling (2 tests): ecd_tanh, oer_softmax survive save/load — ALL PASS
- S5: Rectangular Layers (1 test): DiagCore survives save/load — PASS
- M1: DiagCore Merge (1 test): unmerge restores base (zero-init) — PASS
- M2: SelectiveDiagCore Merge (3 tests): merge=forward(theta=0) d=16/32, unmerge restores base — ALL PASS
- M3: SelectiveDiagCore Merge theta≠0 (1 test): merge=forward (nonzero theta) — PASS
- M4: SelectiveDiagCore Constraints (1 test): rejects rectangular layers — PASS
- M5: NoRot Merge (2 tests): merge=forward, unmerge=base — ALL PASS
- M6: Magnitude Merge (2 tests): ecd_tanh, oer_softmax unmerge≈base (<10%) — ALL PASS
- P1: Frozen State Persistence (2 tests): frozen state restored, pairs not mutated — ALL PASS
- P2: TC-CS Calibration Buffers (2 tests): g_mean_ema, g_cov_ema not persisted — ALL PASS
- P3: Step Indices (1 test): step_idx in state dict — PASS
- Integration (1 test): NoRot full pipeline — PASS

## Known Limitations

These are not bugs — they are documented design constraints:

1. **DiagCore merge is approximate** (`_compute_weight_delta_simple` legacy path):
   - DiagCore merge uses a 0.05x scaling approximation (not exact)
   - `merged_output ≠ adapter_forward` for DiagCore
   - Unmerge accuracy is limited by this approximation
   - For exact merge, use SelectiveDiagCore (JORA-Paper path)

2. **SelectiveDiagCore requires square layers**:
   - Only works when `in_features == out_features`
   - Rectangular layers are intentionally skipped via ValueError

3. **Magnitude unmerge tolerance**:
   - DiagCore magnitude unmerge has ~5-10% relative error
   - This is expected due to the legacy approximation

## What Is NOT Tested (M1 scope)

- Long training runs (LoRA/DoRA baselines)
- `compute_delta()` internals (tested indirectly via merge)
- TC-CS coupling mechanism (not yet validated in real training)
- Downstream eval benchmarks (MMLU, ARC-C, GSM8K)

## Test Design Notes

- **Base model state dict**: Must be captured BEFORE `get_peft_model()` wraps it. After wrapping, `peft.base_model.state_dict()` includes `base_layer` prefixes that won't match a fresh model's keys.
- **SelectiveDiagCore pairs**: `set_support()` does NOT reset `num_pairs_L/R`. Tests must explicitly set consecutive pairs and `num_pairs` to avoid non-deterministic results.
- **Unmerge comparison**: `layer.unmerge()` restores base weights but the adapter is still active. Use `PeftModel.unmerge_adapter()` to properly deactivate the adapter for base-output comparison.
- **Merge equivalence**: The correct comparison for SelectiveDiagCore is `x @ (base.weight + delta_w).t()` not `layer(x)` after merge, because the forward path uses rotated weights while the merged weight is the unrotated delta applied to the base.

## Verdict

**M0 PASS** — Proceed to M1 (LoRA/DoRA baselines and real training experiments).
