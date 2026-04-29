# M1 Baseline Configuration Audit

**Date**: 2026-04-28
**Purpose**: Audit existing LoRA/DoRA baseline configs for comparable-param comparability with JORA-Diag before launching M1 training runs.
**Status**: Audit complete. Action required before running baselines.

---

## 1. Base Model Architecture (OPT-350m)

| Property | Value |
|---|---|
| Total params | 331,196,416 |
| Hidden size | 1024 |
| FFN intermediate size | 4096 |
| Num layers | 24 |
| Target modules (`all-linear`) | 147 linear layers |

### All-Linear Target Breakdown (147 modules)

| Module type | Count | Params/module | Total |
|---|---|---|---|
| `q_proj` (per layer) | 24 | 1,048,576 | 25,165,824 |
| `k_proj` (per layer) | 24 | 1,048,576 | 25,165,824 |
| `v_proj` (per layer) | 24 | 1,048,576 | 25,165,824 |
| `out_proj` (per layer) | 24 | 1,048,576 | 25,165,824 |
| `fc1` (MLP, per layer) | 24 | 4,194,304 | 100,663,296 |
| `fc2` (MLP, per layer) | 24 | 4,194,304 | 100,663,296 |
| `project_in` | 1 | 524,288 | 524,288 |
| `project_out` | 1 | 524,288 | 524,288 |
| **Total** | **147** | — | **331,196,416** |

> All configs use `all-linear` as target modules (via HF's `INCLUDE_LINEAR_LAYERS_SHORTHAND`), which maps to all 147 linear layers above.

---

## 2. Trainable Parameter Counts (Actual, via `get_peft_model`)

Measured by instantiating `JoraConfig`/`LoraConfig` + `get_peft_model` on OPT-350m.

| Method | Variant | Trainable Params | vs JORA-Diag ON | % of Base Model |
|---|---|---|---|---|
| **JORA-Diag** | ON (S_L=32, S_R=32) | **157,824** | 1.00× | 0.0476% |
| **JORA-Diag** | NoRot (S_L=0, S_R=0) | **148,480** | 0.94× | 0.0448% |
| **LoRA** | r=1, alpha=2 | 445,440 | **2.82×** | 0.1344% |
| **LoRA** | r=2, alpha=4 | 890,880 | **5.64×** | 0.2689% |
| **DoRA** | r=1, alpha=2 | 668,160 | **4.23×** | 0.2016% |
| **DoRA** | r=2, alpha=4 | 1,113,600 | **7.05×** | 0.3361% |

### JORA-Diag ON Breakdown (157,824 total)

| Param type | Count | Formula |
|---|---|---|
| `theta_L` | 4,672 | 2 × S_L × N_target_layers (all 146 targeted modules) |
| `theta_R` | 4,672 | 2 × S_R × N_target_layers |
| `diag_params` | 148,480 | m × N_target_layers (m=1024 for attn, 4096 for FFN) |
| **Total** | **157,824** | |

> JORA-Diag ON: 157,824 = 128 × 146 + 9,344. The diag_params count (148,480) accounts for both 1024-dim attention layers (24×4=96 modules) and 4096-dim FFN layers (24×2=48 modules), plus 2 projection layers (512-dim each).

---

## 3. Existing Baseline Config Audit

### 3.1 Config Comparison Table

| Parameter | JORA-Diag ON (main) | JORA-NoRot (baseline) | LoRA r=1 | LoRA r=2 | DoRA r=1 (missing) |
|---|---|---|---|---|---|
| Config file | `run_diag_main_s42_3ep.json` | `run_diag_no_rotation_s42_3ep.json` | `run_lora_baseline.json` | `run_lora_r2_baseline.json` | **MISSING** |
| Model | facebook/opt-350m | facebook/opt-350m | facebook/opt-350m | facebook/opt-350m | facebook/opt-350m |
| Dataset | yahma/alpaca-cleaned | yahma/alpaca-cleaned | yahma/alpaca-cleaned | yahma/alpaca-cleaned | — |
| Target modules | all-linear | all-linear | all-linear | all-linear | — |
| Max length | 256 | 256 | 256 | 256 | — |
| Epochs | 3 | 3 | **1** | **1** | — |
| Batch size | 4 | 4 | 4 | 4 | — |
| Grad accum | 4 | 4 | 4 | 4 | — |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 | — |
| LR scheduler | constant | constant | constant | constant | — |
| Weight decay | 0.0 | 0.0 | 0.0 | 0.0 | — |
| Warmup ratio | 0.0 | 0.0 | 0.0 | 0.0 | — |
| Max grad norm | 1.0 | 1.0 | 1.0 | 1.0 | — |
| bf16 | true | true | true | true | — |
| Seed | 42 | 42 | 42 | 42 | — |
| **Trainable params** | **157,824** | **148,480** | **445,440** | **890,880** | **MISSING** |
| **Comparable-param?** | — | YES (same method) | NO (2.82× more) | NO (5.64× more) | — |
| **Epochs matched?** | YES | YES | NO (1ep vs 3ep) | NO (1ep vs 3ep) | — |

### 3.2 Key Findings

#### Finding 1 — LoRA configs NOT comparable-param
- **LoRA r=1** has **445,440** params vs JORA-Diag's **157,824** — 2.82× more.
- **LoRA r=2** has **890,880** params — 5.64× more.
- These are NOT comparable-param baselines. They are higher-capacity LoRA, not fair parameter-budget comparisons.
- Root cause: LoRA's `2 × r × d` per-module cost means r=1 gives 2×1024=2048 params/module. For 146 modules: 2048×146=299,008... but due to 512-dim projection layers: 445,440 total.

#### Finding 2 — DoRA config does NOT exist
- No DoRA config file exists in `configs/`.
- DoRA is supported by `train.py` (`use_dora` flag) and the `LoraConfig(use_dora=True)` code path.
- A new config must be created.

#### Finding 3 — Epoch mismatch
- LoRA configs use `num_train_epochs: 1`, while JORA-Diag main uses **3 epochs**.
- LoRA baselines CANNOT be directly compared to JORA-Diag 3ep results without re-running at 3 epochs.
- For fair comparison, either: (a) re-run LoRA at 3 epochs, or (b) re-run JORA at 1 epoch.

#### Finding 4 — Trainable param count DISCREPANCY with prior reference
- The terminal transcript referenced "157,824" for JORA-Diag and "445,440" for LoRA-r1.
- This is confirmed accurate: JORA-Diag ON = 157,824, LoRA r=1 = 445,440.
- The "445,440" was previously described as "LoRA-r1" — but 445,440 corresponds to LoRA r=1, not r=2.
- **Wait**: LoRA r=1 gives 445,440. LoRA r=2 gives 890,880. The existing config `run_lora_r2_baseline.json` with r=2 gives 890,880.
- So the names in existing configs ARE correct (r=1 and r=2 respectively).
- But: `run_lora_baseline.json` (r=1) = 445,440 is 2.82× JORA-Diag. This is NOT comparable-param.

#### Finding 5 — LoRA alpha/param relationship
- LoRA r=1, alpha=2: params = 2×1×1024×6×24 + extra for 512-dim = 445,440.
- LoRA r=2, alpha=4: 2×2×1024×6×24 + extra = 890,880.
- The alpha values (2 and 4) are standard LoRA conventions (alpha=2r). These don't affect param count — only scaling.

---

## 4. Comparable-Param Analysis

### The Math

For OPT-350m with `all-linear` targets (146 targeted modules):
- **LoRA**: `2 × r × d` per module, so `2 × r × 1024 × 6 × 24 + extras = 294,912×r`
  - r=1 → 445,440 (actual), not 294,912 (analytical) — discrepancy due to FFN 4096-dim modules
  - True: 445,440 = r×(attn: 2×1024×96 + ffn: 2×4096×48 + proj: 2×512×2) = r×445,440
- **JORA-Diag ON**: 157,824 (theta=9,344 + diag=148,480)

### Can we make LoRA comparable to JORA-Diag?

To match JORA-Diag's 157,824 params, LoRA would need:
- `r × 445,440 = 157,824` → `r = 0.35` — not a valid integer rank.

**Implication**: JORA-Diag achieves ~157K params because its diagonal core reuses the base layer's full dimensionality without expanding to a low-rank bottleneck. LoRA's rank-r bottleneck cannot achieve this parameter budget at the same layer count.

**Conclusion**: True comparable-param is impossible. The best achievable is:
- **LoRA r=1** at 445,440 (2.82× JORA-Diag) as the "lower capacity" LoRA baseline.
- **LoRA r=2** at 890,880 (5.64× JORA-Diag) as the "higher capacity" LoRA baseline.
- Accept the param asymmetry and document it as a known limitation.

### Alternative framing

Instead of comparable-param, frame baselines as:
1. **Efficiency comparison**: JORA-Diag achieves comparable training quality to LoRA at ~35% of the trainable parameters (157K vs 445K at r=1).
2. **Quality comparison at matched training**: JORA-Diag vs LoRA at matched dataset, epochs, and training setup (accept param asymmetry).

---

## 5. Matched vs Non-Matched Classification

| Config | Status | Issue |
|---|---|---|
| JORA-Diag ON (3ep) | Reference | Main method |
| JORA-NoRot (3ep) | **Matched** | Same epochs, same dataset, same seed — for mechanism claim |
| LoRA r=1 | **Epoch mismatch** | 1ep vs 3ep; need re-run or JORA at 1ep |
| LoRA r=2 | **Epoch mismatch** | 1ep vs 3ep; need re-run or JORA at 1ep |
| DoRA r=1 | **Missing** | Config doesn't exist; must create |

### Required: Epoch Alignment Decision

Two options:

**Option A — Re-run JORA at 1 epoch for LoRA comparison**
- Run: JORA-Diag ON at 1ep (matched with LoRA's current 1ep)
- Run: JORA-NoRot at 1ep
- Pros: Quick; directly comparable with existing LoRA 1ep data
- Cons: 1ep evidence already exists (R004/R005), but 3ep is more converged

**Option B — Re-run LoRA at 3 epochs**
- Run: LoRA r=1 at 3ep
- Run: LoRA r=2 at 3ep
- Run: DoRA r=1 at 3ep
- Pros: Matches JORA-Diag's 3ep fully
- Cons: ~3× longer per run; 3 runs needed

**Recommendation**: **Option A** (1ep comparison) is faster and sufficient for an initial paper-quality baseline. Use existing 1ep JORA data (R004/R005) + new 1ep LoRA runs. The 3ep data (E1/E2) is the main quality evidence; the 1ep LoRA comparison is for relative standing context.

---

## 6. New Configs Required

### Config A: `configs/run_lora_r1_3ep.json` (NEW — recommended)
For 3ep comparison with JORA-Diag ON.

```json
{
  "_description": "LoRA r=1, 3ep. Matched epochs to JORA-Diag ON for paper-quality baseline.",
  "_method": "LoRA-r1-3ep",
  "_compares_to": "JORA-Diag ON (run_diag_main_s42_3ep.json)",
  "_note": "2.82x more trainable params than JORA-Diag ON. Accept param asymmetry.",

  "model_name_or_path": "facebook/opt-350m",
  "dataset_name": "yahma/alpaca-cleaned",
  "splits": "train",
  "max_length": 256,
  "num_train_epochs": 3,
  "logging_steps": 20,
  "log_level": "info",
  "logging_strategy": "steps",
  "eval_strategy": "no",
  "save_strategy": "no",
  "bf16": true,
  "packing": false,
  "learning_rate": 1e-4,
  "lr_scheduler_type": "constant",
  "weight_decay": 0.0,
  "warmup_ratio": 0.0,
  "max_grad_norm": 1.0,
  "output_dir": "results/run_lora_r1_3ep_s42",
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": false,
  "dataset_text_field": "text",
  "use_peft_lora": true,
  "lora_r": 1,
  "lora_alpha": 2,
  "lora_target_modules": "all-linear",
  "seed": 42
}
```

### Config B: `configs/run_dora_r1_3ep.json` (NEW — required)
For DoRA comparison.

```json
{
  "_description": "DoRA r=1, 3ep. Matched epochs to JORA-Diag ON.",
  "_method": "DoRA-r1-3ep",
  "_compares_to": "JORA-Diag ON (run_diag_main_s42_3ep.json), LoRA r=1 3ep",
  "_note": "4.23x more trainable params than JORA-Diag ON. DoRA = LoRA + magnitude decomposition.",

  "model_name_or_path": "facebook/opt-350m",
  "dataset_name": "yahma/alpaca-cleaned",
  "splits": "train",
  "max_length": 256,
  "num_train_epochs": 3,
  "logging_steps": 20,
  "log_level": "info",
  "logging_strategy": "steps",
  "eval_strategy": "no",
  "save_strategy": "no",
  "bf16": true,
  "packing": false,
  "learning_rate": 1e-4,
  "lr_scheduler_type": "constant",
  "weight_decay": 0.0,
  "warmup_ratio": 0.0,
  "max_grad_norm": 1.0,
  "output_dir": "results/run_dora_r1_3ep_s42",
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": false,
  "dataset_text_field": "text",
  "use_peft_lora": true,
  "use_dora": true,
  "lora_r": 1,
  "lora_alpha": 2,
  "lora_target_modules": "all-linear",
  "seed": 42
}
```

---

## 7. Minimal Run Order

Before any LoRA/DoRA comparison run is launched, the following configs must exist and be verified:

### Config creation checklist
- [ ] `configs/run_lora_r1_3ep.json` — create (copy of `run_lora_baseline.json` + num_train_epochs=3)
- [ ] `configs/run_dora_r1_3ep.json` — create (DoRA r=1, 3ep)
- [ ] Verify both configs parse without error (dry run)
- [ ] Verify trainable param counts via `print_trainable_parameters()` (dry run with --max_steps 1)

### Run order (after configs created)
1. **JORA-Diag ON 3ep** — already done (R006, E1). No re-run needed.
2. **JORA-NoRot 3ep** — already done (R007, E2). No re-run needed.
3. **LoRA r=1 3ep** — new run, ~55 min (1 GPU). Compare to E1.
4. **DoRA r=1 3ep** — new run, ~55 min (1 GPU). Compare to E1.

### Optional follow-up (after M1 initial results)
- LoRA r=2 3ep: if DoRA/LoRA r=1 both underperform JORA-Diag, r=2 tests the "more capacity" hypothesis.
- JORA-Diag at 1ep: if LoRA r=1 at 1ep shows interesting behavior, JORA-Diag at 1ep provides matched comparison.

---

## 8. Summary: Audit Verdict

| Item | Status | Action |
|---|---|---|
| Existing LoRA r=1 config | **Epoch mismatch** | Create `run_lora_r1_3ep.json` |
| Existing LoRA r=2 config | **Epoch mismatch** | Optional: create 3ep version later |
| DoRA config | **Missing** | Create `run_dora_r1_3ep.json` |
| Target modules alignment | **OK** | All configs use `all-linear` |
| Learning rate alignment | **OK** | All configs use 1e-4 constant |
| Batch/accum alignment | **OK** | All configs use batch=4, accum=4 |
| Seed alignment | **OK** | All configs use seed=42 |
| Param count comparability | **Asymmetric** | Document: LoRA 2.82×, DoRA 4.23× more params |
| Trainability correctness | **OK** | DiagCore NoRot (148,480) is diagonal-only, no rotation params |

### Recommendation

**Do NOT launch existing LoRA configs** (they are 1ep and cannot be compared to JORA-Diag 3ep). Create the two new 3ep configs above and launch those.

The key comparison for the paper is: **Does JORA-Diag ON at 3ep achieve comparable or better train loss than LoRA r=1 at 3ep?** If yes (with 2.82× fewer params), JORA-Diag's efficiency story holds. If no, the paper must acknowledge that the efficiency advantage does not translate to quality parity.

---

## 9. Evidence Needed from M1 Runs

After LoRA r=1 3ep and DoRA r=1 3ep complete, the minimum comparison table for the paper is:

| Config | Final train_loss | Token accuracy | Runtime | Trainable params |
|---|---|---|---|---|
| JORA-Diag ON 3ep | E1 | E1 | E1 | 157,824 |
| JORA-NoRot 3ep | E2 | E2 | E2 | 148,480 |
| LoRA r=1 3ep | **NEW** | **NEW** | **NEW** | 445,440 |
| DoRA r=1 3ep | **NEW** | **NEW** | **NEW** | 668,160 |

Interpretation guidelines:
- If JORA-Diag train_loss < LoRA r=1 train_loss: JORA-Diag beats LoRA at 35% of the params.
- If JORA-Diag train_loss ≈ LoRA r=1: Parity at 35% of params is a strong efficiency claim.
- If JORA-Diag train_loss > LoRA r=1: Acknowledge the quality gap; efficiency story weakens.

> Note: Train loss comparison is necessary but not sufficient for a paper. M2 (downstream eval) is required for any quality claim.
