# Result-to-Claim: Current State (2026-04-28)

**Purpose**: Adjudicate which claims the current evidence supports, which it forbids, and what the minimum remaining work is before paper writing.

**Evidence date**: R006/R007 3ep run (seed 42), TC-CS Steps 4.5–4.8, M1 LoRA/DoRA baselines (2026-04-28).

---

## 1. Evidence Table

| # | Evidence ID | System | Split | Key metrics | Verdict |
|---|-----------|--------|-------|-------------|---------|
| E1 | R006 | JORA-Diag ON (S_L=32, S_R=32) | train, 3ep, seed 42 | train_loss=2.23682, token_acc=0.51471, runtime=9473s (~158 min), step_time=1.027s | Partial |
| E2 | R007 | JORA-NoRot (S_L=0, S_R=0) | train, 3ep, seed 42 | train_loss=2.23737, token_acc=0.51498, runtime=3315s (~55 min), step_time=0.359s | Done |
| E3 | R004/R005 | JORA-Diag ON vs NoRot | train, 1ep, seed 42 | loss delta ~4e-5, runtime ON=3x OFF | Done |
| E4 | Step 4.5 | TC-CS-1S vs Consecutive checkpoint analysis | post-hoc | pair overlap = 100% (384/384 pairs), 0 dimensions differ | FAIL |
| E5 | Step 4.6 | Toy/random matrix overlap test | offline | normalized_corr vs energy_product = 4.2% overlap | PASS (toy) |
| E6 | Step 4.7 | grad_col_ema rank-1 analysis + pool sensitivity | offline | outer(gc) rank-1 = 100.0%, corr(outer, energy_product) = 1.0000, all pool sizes >= 80% overlap | FAIL (root cause) |
| E7 | Step 4.8 | TC-CS-1S matched training (100 steps) | train, 100 steps | train_loss delta = 2.7e-6 (noise), 0 mechanism differentiation | FAIL |
| E8 | M1 LoRA | LoRA r=1, 3ep, seed 42 | train, 3ep, seed 42 | train_loss=1.9538, token_acc=0.5675, runtime=2484s (~41 min) | Done |
| E9 | M1 DoRA | DoRA r=1, 3ep, seed 42 | train, 3ep, seed 42 | train_loss=1.9455, token_acc=0.5708, runtime=4476s (~75 min) | Done |

### Key deltas (E1 vs E2)

| Metric | JORA-Diag ON | JORA-NoRot | Delta (ON − NoRot) | Interpretation |
|--------|-------------|------------|--------------------|---------------|
| Final train loss | 2.23682 | 2.23737 | **−0.00055** | ON slightly lower; noise-level |
| Mean token accuracy | 0.51471 | 0.51498 | **−0.00027** | NoRot slightly higher; noise-level |
| Runtime (min) | 157.9 | 55.2 | **+102.6** | ON is 2.86× slower |
| Step time (s) | 1.027 | 0.359 | **+0.668** | ON is 2.86× slower |
| Total steps | 9222 | 9222 | 0 | Fully matched |

### Key deltas — JORA vs LoRA/DoRA (E1/E2 vs E8/E9)

|| Metric | JORA-Diag ON | LoRA r=1 (E8) | DoRA r=1 (E9) | Interpretation |
|--------|-------------|-----------|----------|---------------|
|| Final train loss | **2.23682** | **1.95380** | **1.94550** | LoRA/DoRA significantly lower (better) |
|| Mean token accuracy | 0.51471 | 0.56751 | 0.57078 | LoRA/DoRA ~5% higher |
|| Runtime (min) | 157.9 | 41.4 | 74.6 | LoRA fastest; JORA slowest |
|| Trainable params | **157,824** | 445,440 | 668,160 | JORA most parameter-efficient |
|| **train_loss gap vs JORA** | — | **−0.283** | **−0.291** | **LoRA/DoRA >> JORA (major gap)** |

> **Critical finding (2026-04-28)**: JORA-Diag (157,824 params) converges to train_loss ~2.24 after 3ep, while LoRA r=1 (445,440 params) converges to ~1.95 — a gap of 0.28. This is not a small deficit. LoRA/DoRA achieve both better quality AND faster runtime. The efficiency advantage of JORA-Diag (3× fewer params) does not compensate for this quality gap. This finding severely challenges the paper's viability as a competitive-method story.

---

## 2. Supported Claims

The following claims are allowed under the current evidence:

### Claim S1 — JORA-Diag is the current main method
**Evidence**: Code/config repo structure, `AGENTS.md`, `JORA_RESEARCH_CONTRACT.md`.
**Allowed wording**: "JORA-Diag is the current mainline method in this implementation, configured as a structured additive diagonal adapter with sparse rotation basis reparameterization."

### Claim S2 — Diagonal core is the effective capacity source
**Evidence**: E1–E3. Rotation ON and NoRot are nearly identical in quality at both 1ep and 3ep. ON adds runtime cost with no commensurate gain.
**Allowed wording**: "Current evidence suggests the diagonal core is the main effective adaptation capacity in JORA-Diag. Sparse rotations do not show a clear independent benefit under matched training."

### Claim S3 — JORA-NoRot is a first-class mechanism baseline
**Evidence**: E2, E3. NoRot is S_L=S_R=0 (identity rotation). It is an exact subset of JORA-Diag.
**Allowed wording**: "JORA-NoRot is used as the claim-determining mechanism baseline throughout this work."

### Claim S4 — Rotation ON shows no quality advantage over NoRot at matched 1ep and 3ep
**Evidence**: E1, E2, E3. At 3ep: loss delta = −0.00055 (ON better), token_acc delta = −0.00027 (NoRot better). Both differences are noise-level. Runtime is 2.86× worse for ON.
**Allowed wording**: "Under matched training (same model, dataset, horizon, seed), JORA-Diag with rotation ON and JORA-NoRot (rotation OFF) produce nearly identical final train loss and token accuracy. Rotation ON incurs a ~2.86× runtime overhead."

### Claim S5 — TC-CS attempt failed its mechanism gate
**Evidence**: E4, E6, E7. Pair overlap = 100% (384/384 pairs). Root cause: `grad_col_ema` outer product is exactly rank-1 by construction, making `|outer(gc)|` and `|gc[i]*gc[j]|` identical matrices. Step 4.6 toy PASS (4.2%) does not transfer to real activations (100% overlap).
**Allowed wording**: "An attempt to restrict rotation to a task-conditioned coupling subspace (TC-CS) was evaluated but failed to differentiate from energy-based consecutive pairing in real activation regimes. The coupling signal collapsed to the same attractor as energy-based selection."

### Claim S6 — JORA-Diag forward is additive, not residualized
**Evidence**: `docs/FORMULA_AUDIT.md`, code inspection. `Δ(x) = R_L^T Diag(d) R_R x`. No `−x` residualization term.
**Allowed wording**: "JORA-Diag uses an additive delta: the base residual `W_0 x` handles identity; the adapter outputs `Δ(x)` additively."

---

## 3. Forbidden Claims

The following claims are not supported by current evidence and are forbidden under the research contract:

| # | Forbidden claim | Reason |
|---|----------------|--------|
| F1 | "Rotation drives the gain" | E1–E3: ON ≉ NoRot; ON is slower with no quality benefit |
| F2 | "JORA-Diag is validated over JORA-NoRot" | E1–E3: metrics are within noise; no positive evidence |
| F3 | "TC-CS rotation revival succeeded" | E4–E7: 100% pair overlap; training delta = 2.7e-6; gate FAIL |
| F4 | "TC-CS coupling subspace meaningfully differs from consecutive selection" | E6: `outer(gc)` is rank-1 by construction; they are the same ordering |
| F5 | "The main method is the paper-exact residualized full-support operator" | Code uses additive DiagCore; `Δ(x) = R_L^T Diag(d) R_R x`, not `R_L^T Diag(1+d) R_R x − x` |
| F6 | "Rotation is worth its runtime cost" | E1 vs E2: 2.86× runtime overhead; no quality offset |
| F7 | "Exact mergeability holds uniformly across the JORA family" | Only SelectiveDiagCore has verified exact merge; DiagCore uses approximate merge |
| F8 | "JORA-Diag is competitive with LoRA/DoRA at comparable or better quality" | E8/E9: JORA train_loss ~2.24 vs LoRA/DoRA ~1.95 — 0.28 gap; LoRA also 2.86× faster |
| F9 | "Residualized DiagCore is a viable operator for the main JORA-Diag method" | C2.1: catastrophic failure (loss 15.7 vs base 5.5); R_L^T @ R_R ≠ I when pairs are independent; operator fundamentally unsuitable without a projector constraint |

---

## 4. Paper Narrative Update

### Core narrative (post-TC-CS failure)

**If written today**, the paper narrative must be:

> JORA-Diag is a structured additive diagonal adapter for LLM fine-tuning. Its diagonal core provides the main adaptation capacity; sparse rotations are an optional basis reparameterization whose independent contribution is not supported by matched 1ep and 3ep evidence. The method is evaluated against JORA-NoRot (diagonal-only baseline) and against comparable-parameter LoRA/DoRA baselines.

### What TC-CS becomes in the paper

- **Move to future work / negative result section**: "We attempted task-conditioned coupling subspace rotation to give rotation a unique geometric role. This attempt failed to produce mechanism differentiation on real activations (Section X, Appendix Y)."
- **Not a claimed contribution**: TC-CS is documented as a negative result, not a method.
- **Not in the abstract**: The abstract should describe JORA-Diag as additive diagonal + optional rotation basis, not as a rotation-driven method.

### What JORA-NoRot becomes in the paper

- **First-class baseline**: Appears in all mechanism comparisons.
- **Framing**: "JORA-NoRot establishes the diagonal-core-only performance floor. JORA-Diag with rotation ON does not clearly exceed this floor within matched training evidence."

### What rotation becomes in the paper

- **In the Method section**: Described as sparse Givens-parameterized orthogonal transforms as basis reparameterization. Explicitly labeled as optional.
- **In experiments**: Shown as a runtime cost with no demonstrated benefit under current evidence.
- **In the narrative**: Acknowledged honestly rather than papered over.

### Option C as negative result

We attempted a full-support residualized orthogonal correction path (`Δ(x) = R_L^T (I+Diag(d)) R_R x - x`) as Option C to align DiagCore with the SelectiveDiagCore paper-exact formula. This failed catastrophically: 1ep eval loss = 15.7 vs base 5.5 (vs additive JORA-Diag 2.24). Root cause: independent left/right rotation pairs mean `R_L^T @ R_R ≠ I` at init. The negative result is documented in `docs/JORA_OPTION_C_POSTMORTEM.md`.

---

## 5. Minimal Remaining Experiments Before Writing

The following are the minimum experiments that must be completed before paper drafting:

### M0 — Deployment sanity (required for any paper)
**Status: PASS (27/27 tests, 2026-04-28)**
| Check | Status | Notes |
|-------|--------|-------|
| JORA-NoRot save/load roundtrip | PASS | Output identity, theta=None preserved, merged≈base |
| JORA-Diag save/load roundtrip | PASS | Output identity, theta preserved, pairs preserved |
| SelectiveDiagCore merge | PASS | Exact by basis probing — only variant with verified exact merge |
| DiagCore merge | PASS (approximate) | `_compute_weight_delta_simple` legacy path; 0.05x scaling approximation; NOT exact |
| NoRot merge/unmerge | PASS | Exact via `test_diag_core_unmerge_equals_original` |
| Magnitude variants (ecd_tanh, oer_softmax) | PASS | Survive save/load; unmerge has ~5-10% relative error |

**Key distinction**: Exact merge is only verified for `SelectiveDiagCore`. `DiagCore` (JORA-Diag mainline) uses an approximate legacy merge path and cannot be claimed as exact.

### M1 — LoRA / DoRA comparable-param baselines (required for competitive claim)

**Status: COMPLETED with critical finding. Training runs done. Analysis complete.**

| Run | Priority | Purpose | Status |
|-----|---------|---------|--------|
| LoRA r=1, 3ep | **MUST** | Establishes whether JORA-Diag is competitive with LoRA | COMPLETED. train_loss=1.9538, token_acc=0.5675, runtime=41 min. **FAILED to beat JORA.** |
| DoRA r=1, 3ep | **MUST** | Establishes whether JORA-Diag is competitive with DoRA | COMPLETED. train_loss=1.9455, token_acc=0.5708, runtime=75 min. **FAILED to beat JORA.** |
| Parameter count audit | **MUST** | Document exact counts and limitations | DONE. See `experiments/M1_BASELINE_CONFIG_AUDIT.md`. |

**Critical finding**: LoRA r=1 and DoRA r=1 both achieve train_loss ~1.95 after 3ep, while JORA-Diag ON converges to ~2.24. The gap is 0.28 — a very large deficit. JORA-Diag also has the slowest runtime (158 min) among all four methods. The efficiency story (35% of LoRA's params) does not compensate for the quality and speed gap.

**Implication for paper**: The current JORA-Diag method cannot claim competitive quality or efficiency vs LoRA/DoRA. A fundamental rethink of the approach is required before the paper can proceed.

### M2 — Downstream evaluation (required for paper credibility)
| Eval | Priority | Purpose |
|------|---------|---------|
| MMLU (5-shot) | **MUST** | Standard academic benchmark |
| ARC-C (challenge) | **MUST** | Standard academic benchmark |
| GSM8K | NICE | If time/compute permits |
| Hellaswag | NICE | If time/compute permits |

**Note**: E1/E2 are training-side metrics only. A paper requires held-out evaluation to make any quality claim.

### M3 — Narrative-freeze decision
Before writing begins, confirm:
- [ ] TC-CS status is documented as failed mechanism gate (negative result)
- [ ] JORA-Diag narrative is centered on additive diagonal + optional rotation basis
- [ ] JORA-NoRot is listed as first-class baseline in all comparisons
- [ ] Research contract (docs/JORA_RESEARCH_CONTRACT.md) is updated to reflect TC-CS failure
- [ ] AGENTS.md reflects current verdict

---

## 6. Option C (Residualized DiagCore) — FAILED (2026-04-28)

**Gate**: C2.0b gradient sanity + C2.1 1ep matched training.

**C2.0b**: PASS. Gradients flow to `diag_params`, `theta_L`, and `theta_R` at every step.

**C2.1**: **CATASTROPHIC FAIL**. Trained residualized ON and NoRot for 1ep on opt-350m/alpaca-cleaned.

| Model | Eval Loss | vs Base | Verdict |
|---|---|---|---|
| Base model | 5.4572 | — | — |
| Additive JORA-Diag ON | 2.2387 | -3.22 | Correct learning |
| **Residualized ON** | **15.7185** | **+10.26** | **Catastrophic** |
| **Residualized NoRot** | **19.1891** | **+13.73** | **Catastrophic** |

**Root cause**: For DiagCore (full-support), `pairs_L` and `pairs_R` are independently sampled. So `R_L^T @ R_R = O ≠ I` even at theta=0. The residualized form `Δ(x) = R_L^T (I+Diag(d)) R_R x - x` does NOT give zero-function-change at init. During training, theta drifts large (max=1.67) and the residualized adapter catastrophically reshapes hidden-state geometry.

SelectiveDiagCore is immune because it uses projector `P_U` (satisfies `P_U = P_U^T @ P_U`), making `R_L^T @ P_U @ R_R = P_U` at theta=0, giving exact zero-function-change.

**Action**: DiagCore reverted to additive default. Residualized path kept only for SelectiveDiagCore. See `docs/JORA_OPTION_C_POSTMORTEM.md`.

---

## 7. Summary Verdict

| Question | Answer | Evidence |
|----------|--------|---------|
| Does rotation add value? | **No** (under current evidence) | E1–E3: ON ≈ NoRot at 1ep and 3ep; ON 2.86× slower |
| Is TC-CS a viable mechanism? | **No** | E4–E7: 100% pair overlap; rank-1 collapse; 2.7e-6 training delta |
| What is the main effective component? | **Diagonal core** | E1–E3: NoRot (diagonal-only) ≈ JORA-Diag (diagonal + rotation) |
| Is JORA-Diag competitive with LoRA/DoRA? | **No** (major quality gap) | E8/E9: JORA train_loss=2.24 vs LoRA=1.95 vs DoRA=1.95; gap=0.28. JORA also slowest runtime. |
| Is the paper viable as a competitive-method story? | **Severely challenged** | No quality advantage, no efficiency advantage when quality is accounted for. May require repositioning around mergeability or structure. |
| What is the main paper story? | **Additive diagonal PEFT** | Rotation demoted; diagonal core is the honest mechanism |
| Is TC-CS recoverable? | **Not with current infrastructure** | Requires fundamentally different candidate-pool/score |
| What is the recommended path? | **Option C with strict gates** | See `docs/JORA_PIVOT_OPTIONS.md`. Validate redesign offline before training. |
---

## 7. Project Direction Decision

The triple failure (rotation null, TC-CS failed, JORA vs LoRA gap 0.28) requires a project-level decision before further resource commitment. See `docs/JORA_PIVOT_OPTIONS.md` for full analysis of Option A/B/C.

**Recommendation: Option C with strict gates** — validate the operator redesign offline before any further training runs.

**Immediate gate (C1)**: Offline test that residualized form + non-zero init produces reasonable outputs at init. If this fails, the operator itself is insufficient.
