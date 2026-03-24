# Round 9 Refinement

## Problem Anchor
*(Verbatim from round 0, with targeted corrections — see Anchor Check)*

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope) beats LoRA-r1 (must-win); is competitive with LoRA-r2 within 2pp average (primary); optionally matches LoRA-r2 (stretch). Beats faithful qGOFT at equal budget.

---

## Anchor Check

- **Original bottleneck**: Static rotation structure in orthogonal PEFT wastes capacity; LoRA lacks geometric bias.
- **Why the revised method still addresses it**: Unchanged. This round fixes three precision issues in Algorithm 1 (calibration pseudocode consistency, δ storage form, claim 2 conditionality). No mechanism change.
- **Reviewer suggestions rejected as drift**: None.

---

## Simplicity Check

- **Dominant contribution after revision**: Unchanged from round 8.
- **Changes this round**: Three surgical precision fixes to Algorithm 1 and claims framing. No new components, no new claims.
- **Why the mechanism is still the smallest adequate route**: Identical to round 8.

---

## Changes Made

### 1. Fix Algorithm 1 calibration pseudocode (IMPORTANT — targets Method Specificity)

- **Reviewer said**: "Cannot collect `g = autograd(loss, y_base)` inside `torch.no_grad()`. `task_loss(y_base)` reads like a local-layer loss rather than a full-model loss."
- **Action**: Rewrote the calibration block as:
  - Full-model forward+backward, no `torch.no_grad()` wrapper.
  - `ema_out` collected via a registered backward hook on the q_proj/o_proj output tensor, accumulating `(∂L/∂y)²` per-coordinate.
  - `ema_in` collected via a registered forward hook on the q_proj/o_proj input tensor, accumulating `x²` per-coordinate.
  - No optimizer step is called (just `loss.backward()`, then `optimizer.zero_grad()`).
  - Hooks are deregistered after calibration. Adapter modules are not yet instantiated.
- **Reasoning**: This is literally implementable: register_forward_hook + register_full_backward_hook, accumulate EMA, deregister, allocate adapter. Standard PyTorch pattern.
- **Impact**: Method Specificity should cross the 9/10 threshold.

### 2. Specify δ storage form (IMPORTANT — targets Method Specificity)

- **Reviewer said**: "Specify the implementation form of δ on U unambiguously."
- **Action**: δ is stored as a **packed vector** `δ ∈ R^{|U|}` (not a full-length masked tensor). Application uses `scatter_add` / `gather`:
  ```
  # Forward: apply δ to x̂ on support U
  x̂_U = x̂[U]                    # gather, shape [|U|]
  x̂_U_scaled = (1 + δ) * x̂_U   # elementwise, shape [|U|]
  x̂_out = x̂.clone()
  x̂_out[U] = x̂_U_scaled         # scatter back
  ```
  Memory: `|U| ≤ 128` scalars = 512 bytes. Gradient flows through `x̂[U]` gather correctly.
- **Reasoning**: Packed storage is unambiguous, memory-minimal, and standard. A full-length masked tensor would also work but is 32× larger and less explicit about support.
- **Impact**: Method Specificity — no remaining implementation ambiguity.

### 3. Conditionalize claim 2 on qGOFT reproduction (MINOR — targets Venue Readiness)

- **Reviewer said**: "State explicitly that claim 2 is only made if faithful qGOFT reproduction succeeds."
- **Action**: Added one explicit sentence to the claim 2 block and the experiment handoff: "Claim 2 (JORA beats qGOFT at equal budget) is conditional on successful faithful qGOFT reproduction. If reproduction fails QA checks (parameter count mismatch > 5%, or training divergence), claim 2 is retitled to 'JORA beats fixed-slot JORA at equal budget' and qGOFT is moved to appendix." This is a clean, conditional narrative — not a weakening.
- **Impact**: Venue Readiness improves; reviewers cannot object to unverified baseline claims.

---

## Revised Proposal

# Research Proposal: JORA — Adaptive Sparse Rotation-Slot Allocation for Extreme-Budget Square-Layer PEFT

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope) beats LoRA-r1 as a must-win; is competitive with LoRA-r2 within 2pp (primary headline if achieved); matches LoRA-r2 (stretch if achieved). Beats faithful qGOFT at equal budget (conditional on reproduction).

## Technical Gap

`qGOFT` gives rotation-based PEFT a strong geometric primitive, but its rotation budget is **static**. `LoRA` allocates low-rank capacity without rotation structure. The gap: **within rotation-based PEFT, no method uses early task statistics to allocate a fixed sparse bilateral rotation budget and freezes that support for stable optimization with a zero-function-change initialization**.

## Method Thesis

JORA is a square-layer, mergeable additive adapter that runs a short pre-training calibration phase to score input and output dimensions, allocates **one fixed set of sparse bilateral rotation slots**, and then trains Givens angles plus a residualized selected-support diagonal correction. The allocation is one-time; support is frozen for the remainder of training.

**Core operator** — defined once, used everywhere:

```
D_sel = P_U + diag(δ)_U
```

where:
- `P_U = diag(1_U)` — fixed projection onto U (non-trainable, ones on U, zeros elsewhere)
- `diag(δ)_U` — trainable diagonal correction on U, **initialized to zeros** (zeros elsewhere always)
- `δ` is stored as a **packed vector** `δ ∈ R^{|U|}`, applied via gather/scatter on support indices U

**Forward pass**:
```
x̃    = R_R(slots_R, θ_R) · x
x̂    = D_sel · x̃  =  P_U · x̃  +  diag(δ)_U · x̃   [gather x̃[U], scale by (1+δ), scatter back]
delta = R_L^T(slots_L, θ_L) · x̂  −  P_U · x            ← residualized
out   = base_out + delta
```

At init (θ=0, δ=0): `delta = P_U·x − P_U·x = 0` — zero function change at allocation. All parameters gradient-live from step T_stat.

**Mergeability**: `W_merged = W₀ + R_L^T · D_sel · R_R − P_U` — exact, zero inference overhead.

**Parameter scale** (Mistral-7B, 32 layers, q+o scope):
- JORA-small K=8:  2 × 32 × (8 + 8 + 32)    =  3,072 params  (0.29% of LoRA-r1)
- JORA-base  K=32: 2 × 32 × (32 + 32 + 128)  = 12,288 params  (2.34% of LoRA-r1)
- LoRA-r1 (q+o):   524,288 params
- LoRA-r2 (q+o):   1,048,576 params

**Parameter budget derivation** (why 12,288):
- 32 transformer layers; q_proj and o_proj each d×d, d=4096.
- K=32 Givens pairs per side → K scalar angles per rotation matrix (sparse, O(Kd) matmul).
- Active support: |dims(slots_R)| = 2K=64, |dims(slots_L)| = 2K=64, |U| ≤ 4K=128.
- Per module: θ_R[32] + θ_L[32] + δ[≤128] = 192 params.
- 2 modules/layer × 32 layers = 64 modules × 192 = **12,288 params total**.

## Contribution Focus

- **Single dominant contribution**: One-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, yielding Pareto-dominant extreme-budget accuracy on square attention layers.
- **Explicit non-contributions**: full-width diagonal cores, repeated reseating, broad "adaptive PEFT" claims, rectangular-layer main results, tanh merge path, OER.

## Proposed Method

### Complexity Budget
- **Frozen**: W₀, backbone, tokenizer, PEFT infrastructure; P_U (allocated once, then frozen).
- **New trainable**: θ_R [K], θ_L [K], δ [|U|]. JORA-base (K=32): 192 params/module × 64 modules = 12,288 total.
- **Excluded**: full-width D_diag, repeated reseating, rectangular operators, k/v GQA, MLP, tanh, OER.

### Algorithm 1: JORA — Full Implementation Pseudocode

```
Algorithm 1: JORA (Adaptive Sparse Rotation-Slot Allocation)

Input:
  model          — pretrained Mistral-7B-v0.1 (frozen weights W₀)
  target_modules — {q_proj, o_proj} in each of 32 layers
  K              — number of Givens pairs per side (K=32 for JORA-base)
  T_stat         — calibration steps: min(200, 0.05·T_total)
  β = 0.99       — EMA decay

─── PHASE 0: PRE-TRAINING CALIBRATION ────────────────────────────────────────
# No adapter exists yet. No optimizer step is called.
# Hooks accumulate per-coordinate statistics over the full model.

ema_in  = {m: zeros(d) for m in target_modules}
ema_out = {m: zeros(d) for m in target_modules}

def fwd_hook(m, x, y):          # registered on each target module
    ema_in[m] ← β·ema_in[m] + (1-β)·mean_batch(x[0]²)  # x[0]: input activation

def bwd_hook(m, grad_in, grad_out):  # registered on each target module
    ema_out[m] ← β·ema_out[m] + (1-β)·mean_batch(grad_out[0]²)  # ∂L/∂y

for m in target_modules:
    m.register_forward_hook(fwd_hook)
    m.register_full_backward_hook(bwd_hook)

for t = 1 … T_stat:
    batch ← next(train_loader)
    loss = model(batch).loss        # full-model forward (all layers)
    loss.backward()                 # full-model backward; hooks fire
    optimizer.zero_grad()           # clear gradients, NO optimizer.step()

for m in target_modules:
    m._fwd_hook.remove()
    m._bwd_hook.remove()

# Stability check
for m in target_modules:
    if Jaccard(top_{2K}(ema_in[m]), window=last_5_steps) < 0.85:
        extend T_stat by 50 steps (once)  # re-run loop above

─── PHASE 1: SINGLE ALLOCATION ───────────────────────────────────────────────
for m in target_modules:
    slots_R[m] ← top2K_consecutive_pairs(ema_in[m])   # K disjoint Givens pairs
    slots_L[m] ← top2K_consecutive_pairs(ema_out[m])
    U[m]       ← sorted(dims(slots_R[m]) ∪ dims(slots_L[m]))   # |U| ≤ 4K
    # Packed storage for δ:
    δ[m]   ← zeros(|U[m]|)                # trainable, shape [|U|], packed
    θ_R[m] ← zeros(K)                     # trainable Givens angles, right
    θ_L[m] ← zeros(K)                     # trainable Givens angles, left
    P_U[m] ← U[m]                         # stored as index list, non-trainable

─── PHASE 2: MAIN TRAINING ────────────────────────────────────────────────────
# N optimizer steps (same as all baselines)

def jora_forward(m, x):
    # Step 1: right-side rotation (sparse, O(Kd))
    x̃ = x.clone()
    for k = 1..K:
        i, j = slots_R[m][k]; c, s = cos(θ_R[m][k]), sin(θ_R[m][k])
        x̃[..., i], x̃[..., j] = c*x̃[...,i] - s*x̃[...,j], s*x̃[...,i] + c*x̃[...,j]

    # Step 2: D_sel via gather/scatter (O(|U|))
    x̂ = x̃.clone()
    x̂_U = x̃[..., U[m]]                  # gather, shape [..., |U|]
    x̂[..., U[m]] = (1 + δ[m]) * x̂_U    # scale by (1+δ); scatter back

    # Step 3: left-side rotation transposed (sparse, O(Kd))
    ŷ = x̂.clone()
    for k = K..1:                          # reverse order = R_L^T
        i, j = slots_L[m][k]; c, s = cos(θ_L[m][k]), sin(θ_L[m][k])
        ŷ[..., i], ŷ[..., j] = c*ŷ[...,i] + s*ŷ[...,j], -s*ŷ[...,i] + c*ŷ[...,j]

    # Step 4: residualize (O(|U|))
    delta = ŷ.clone()
    delta[..., U[m]] -= x[..., U[m]]      # subtract P_U·x; zeros outside U
    delta[..., complement(U[m])] = 0      # ensure zero outside U

    # Step 5: base output + adapter delta
    return m.base_forward(x) + delta

# Verify init:
# θ=0, δ=0 → x̃=x, x̂[U]=x[U], ŷ[U]=x[U], delta[U]=x[U]-x[U]=0 ✓

─── PHASE 3: MERGE FOR INFERENCE ──────────────────────────────────────────────
for m in target_modules:
    # Construct R_L^T · D_sel · R_R as a dense matrix (one-time O(Kd²/K)=O(d²) op):
    R_R_dense = apply_givens_sequence(slots_R[m], θ_R[m], eye(d))
    D_sel_dense = diag with (1+δ[m]) on U[m], 0 elsewhere
    R_L_T_dense = apply_givens_sequence_T(slots_L[m], θ_L[m], eye(d))
    delta_W = R_L_T_dense @ D_sel_dense @ R_R_dense
    delta_W[U[m], :] -= eye(d)[U[m], :]   # subtract P_U
    m.weight.data += delta_W               # fuse into W₀; adapter removed
```

**Complexity summary**:

| Operation | Time per module | Extra memory |
|-----------|----------------|--------------|
| Calibration forward+backward | Same as base model (no extra params) | O(d) EMA per module |
| R_sparse (training, per step) | O(Kd) = O(32×4096) ≈ 131K FLOPs | O(d) for x̃ copy |
| D_sel via gather/scatter | O(\|U\|) ≤ O(4K) = O(128) | — |
| Residualization | O(\|U\|) | — |
| δ packed storage | — | \|U\|×4 bytes ≤ 512 bytes/module |
| Optimizer states (AdamW) | — | 2×192×64 modules ≈ 0.1 MB total |
| Inference after merge | Same as base (zero overhead) | — |

Residualization cost: O(128) elementwise ops per module per forward step, vs. O(4096²)=O(16.8M) for the W₀ matmul. Overhead < 0.001%.

### Core Mechanism

#### Operator definition

`D_sel = P_U + diag(δ)_U`

- `P_U`: fixed projection, ones on support dims U, zeros elsewhere. Stored as index list U.
- `diag(δ)_U`: trainable correction, **packed vector δ ∈ R^{|U|}**, zeros at init.
- At δ=0: `D_sel · x̃ = x̃[U]` (selects support dims only).
- At δ≠0: `D_sel · x̃_u = (1 + δ_u) · x̃_u` for u∈U; zero for u∉U.

#### Zero-function-change at allocation

```
init: θ_R = θ_L = 0 → R_R = R_L = I; δ = 0 → D_sel = P_U
  x̃ = x; x̂[U] = x[U]; ŷ[U] = x[U]; delta[U] = x[U] - x[U] = 0 ✓
```

#### Gradient-live at allocation

```
∂delta/∂δ_u |_{init} = x_u  (nonzero for active u∈U) ✓
∂delta/∂θ_R |_{init} ∝ P_U · ∂R_R/∂θ · x  (nonzero for active input dims) ✓
```

#### Asymmetric bilateral statistics

Left side (output) — **gradient signal** (task-coupled), via backward hook:
`ema_out[m][i] ← β·ema_out[m][i] + (1-β)·mean_batch((∂L/∂y_i)²)`

Right side (input) — **activation proxy** (no task coupling), via forward hook:
`ema_in[m][j] ← β·ema_in[m][j] + (1-β)·mean_batch(x_j²)`

Presented honestly as "gradient-guided output allocation + activation-proxy input allocation."

#### Deterministic disjoint pairing rule

1. Stable-sort dims by descending score; ties broken by smaller index first.
2. Take top 2K dims.
3. Pair consecutively: (d₁,d₂), …, (d_{2K-1},d_{2K}).
4. Canonicalize: (min(a,b), max(a,b)).

#### Precommitted Diag-only-selected interpretation rule

Diag-only-selected: same U, no rotations (R_R=R_L=I throughout), only δ trained.

- If JORA − Diag-only-selected > 0.5pp on all three benchmarks: rotation contribution supported; dominant contribution stands.
- If ≤ 0.5pp: paper narrows to "adaptive sparse diagonal scaling with rotation-structured support selection." Still publishable.

This rule is precommitted and claim-determining, not merely diagnostic.

### Training Plan

1. **Backbone / data**: Mistral-7B-v0.1, Alpaca-cleaned 52k, standard SFT.
2. **Target modules**: q_proj, o_proj (32 layers each, 64 modules total).
3. **Calibration phase**: T_stat = min(200, 0.05·T_total) full-model forward+backward steps. No optimizer step. Hooks on target modules accumulate ema_in and ema_out. β=0.99. Wall-clock: ~5 min on 3× RTX 4090. NOT counted in optimizer-step budget.
4. **Stability check**: Jaccard of top-2K sets over last 5 calibration steps. Extend by 50 if < 0.85.
5. **Allocation**: deterministic pairs; init δ=0 (packed), θ=0, U=index list (frozen).
6. **Main training**: N optimizer steps (same as all baselines). AdamW, cosine decay, JORA-specific LR.
7. **Variants**: JORA-small (K=8, ~3K), JORA-base (K=32, ~12K).

### Failure Modes and Diagnostics

- **Warmup unstable (Jaccard < 0.85 after extension)**: report; increase β; flag as limitation.
- **δ explodes**: L2 penalty or clip; detect via max|δ|.
- **Diag-only-selected ≈ JORA**: narrow claim per precommitted rule.
- **qGOFT reproduction fails QA**: retitle claim 2 (see below).
- **JORA-base does not beat LoRA-r1**: paper reports honest null-result Pareto story.

### Limitations

1. **Square-layer restriction**: q_proj and o_proj only. Does not cover rectangular layers (k/v under GQA, MLP).
2. **Calibration overhead**: ~5 min wall-clock per run (not optimizer steps). Non-negligible for very short training runs.
3. **Aggressive empirical bet**: Must-win (JORA-base > LoRA-r1) is the primary validation target. The 2pp-of-LoRA-r2 primary target is conditional on screening results.
4. **Rotation contribution uncertain**: Diag-only-selected may match JORA; narrowing rule is precommitted.
5. **Single dataset in main paper**: Alpaca-cleaned only.

### Novelty and Elegance Argument

- **vs. qGOFT**: same rotation primitive, adaptive vs. static allocation.
- **vs. fixed-slot JORA**: same codepath, same budget, adaptive vs. random — clean isolation.
- **vs. LoRA**: rotation-structured adapter at 2–3% of LoRA-r1 parameter count.
- **Correctness (not novelty)**: residualized init is a necessary correctness property of any adapter introduced mid-training, not a headline contribution.
- **Focused**: one mechanism, one parameter regime, one clear Pareto story.

## Claim-Driven Validation Sketch

### Claim 1 (Primary): JORA Pareto-dominates LoRA in the extreme-budget regime

- **Experiment**: JORA-small (~3K) and JORA-base (~12K) vs. LoRA-r1 (~524K), LoRA-r2 (~1M), LoRA-r4 (~2M).
- **Metric**: average MMLU / ARC-C / GSM8K vs. exact param count.
- **Staged success**: Must-win (JORA-base > LoRA-r1); Primary (within 2pp of LoRA-r2 if achieved); Stretch (≥ LoRA-r2 if achieved).
- **Seeds**: 3 for JORA-base and LoRA-r2; 2 for others.

### Claim 2 (Conditional): JORA beats qGOFT / static baseline at equal budget

- **Condition**: Claim 2 as stated (JORA beats faithful qGOFT) is made only if qGOFT reproduction passes QA: parameter count within 5% of target, training loss converges within 110% of LoRA-r1 loss. If QA fails, claim 2 is retitled to "JORA beats fixed-slot JORA at equal budget" and qGOFT is moved to appendix.
- **Experiment**: JORA-base vs. faithful qGOFT (or fixed-slot JORA as fallback) at matched param count.
- **Metric**: average MMLU / ARC-C / GSM8K.
- **Seeds**: 3 each.

### Mechanism-Isolation (Mandatory)

- **Diag-only-selected**: same U, no rotations, only δ. **Claim-determining** per precommitted rule.
- **Fixed-slot JORA**: same codepath, random slots. Tests allocation policy value.
- **Support stability curves**: Jaccard over calibration phase. Justifies one-shot allocation.
- **Non-residualized vs. residualized init** (appendix): d_sel=1 vs δ=0. Detection: loss spike > 0.1 nats in first 10 post-allocation steps.
- **Seeds**: 2 each.

### Appendix-Only

- DoRA, OFT, BOFT breadth; JORA-large (K=64); calibration length sensitivity (T_stat=100,200,400); non-residualized baseline.

## Experiment Handoff Inputs

- **Must-prove**: (1) JORA-base beats LoRA-r1; (2) JORA-base beats fixed-slot JORA or qGOFT (conditional) at equal budget.
- **Mandatory diagnostics**: Diag-only-selected (claim-determining), fixed-slot JORA, Jaccard curves, init ablation.
- **Critical metrics**: MMLU, ARC-C, GSM8K; exact param counts; mean ± std.
- **Highest-risk assumptions**:
  1. JORA-base (~12K) beats LoRA-r1 (~524K) — core empirical bet
  2. qGOFT reproducible and passes QA (or fallback to fixed-slot JORA)
  3. δ well-conditioned; residualization numerically stable
  4. Diag-only-selected does NOT match JORA — else rotation claim is downgraded

## Compute & Timeline

- **Estimated compute**: ~150 GPU-hours, 3× RTX 4090.
- **Calibration cost**: ~5 min wall-clock per run (negligible).
- **Timeline**: Implementation 1d → qGOFT/fixed-slot 1-2d → 1-seed screening 2d → multi-seed core 5d → analysis+tables 2d.
