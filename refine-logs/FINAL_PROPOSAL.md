# Research Proposal: JORA — Adaptive Sparse Rotation-Slot Allocation for Extreme-Budget Square-Layer PEFT

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope) beats LoRA-r1 as a must-win; within 2pp of LoRA-r2 (primary if achieved); matches LoRA-r2 (stretch if achieved). Beats qGOFT at equal budget (conditional on reproduction).

## Technical Gap

`qGOFT` gives rotation-based PEFT a strong geometric primitive, but its rotation budget is **static**. `LoRA` allocates low-rank capacity without rotation structure. The gap: **within rotation-based PEFT, no method uses early task statistics to allocate a fixed sparse bilateral rotation budget and freezes that support for stable optimization with a zero-function-change initialization**.

## Method Thesis

JORA is a square-layer, mergeable additive adapter that runs a short pre-training calibration phase to score input and output dimensions, allocates **one fixed set of sparse bilateral rotation slots**, and then trains Givens angles plus a residualized selected-support diagonal correction. The allocation is one-time; support is frozen for the remainder of training.

**Core operator**:
```
D_sel = P_U + diag(δ)_U
```
- `P_U = diag(1_U)` — fixed projection onto U (non-trainable, stored as index list)
- `diag(δ)_U` — trainable diagonal correction on U, stored as **packed vector δ ∈ R^{|U|}**, initialized to zeros

**Forward pass**:
```
x̃    = R_R(slots_R, θ_R) · x                                   [O(Kd) sparse Givens]
x̂    = D_sel · x̃  →  x̂[U] = (1 + δ) * x̃[U], x̂[∖U] = 0     [gather/scale/scatter, O(|U|)]
delta = R_L^T(slots_L, θ_L) · x̂  −  P_U · x                   [O(Kd) + O(|U|) residualize]
out   = base_out + delta
```

At init (θ=0, δ=0): `delta = P_U·x − P_U·x = 0`. All parameters gradient-live from T_stat.

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
- **Explicit non-contributions**: full-width diagonal cores, repeated reseating, broad "adaptive PEFT" claims, rectangular-layer main results.

## Proposed Method

### Complexity Budget
- **Frozen**: W₀, backbone, tokenizer, PEFT infrastructure; P_U (allocated once, then frozen).
- **New trainable**: θ_R [K], θ_L [K], δ [|U|]. JORA-base (K=32): 192 params/module × 64 modules = 12,288 total.
- **Excluded**: full-width D_diag, repeated reseating, rectangular operators, k/v GQA, MLP, tanh, OER.

### Algorithm 1: JORA — Full Implementation Pseudocode

```
Algorithm 1: JORA (Adaptive Sparse Rotation-Slot Allocation)

Inputs: model (Mistral-7B, frozen W₀), target_modules={q_proj,o_proj}×32 layers,
        K=32 (Givens pairs), T_stat=min(200,0.05·T_total), β=0.99

─── PHASE 0: CALIBRATION (no optimizer step) ─────────────────────────────────
ema_in[m]  = zeros(d) for m in target_modules
ema_out[m] = zeros(d) for m in target_modules

def fwd_hook(m, inp, out):
    ema_in[m] ← β·ema_in[m] + (1-β)·mean_batch(inp[0]²)   # activation proxy

def bwd_hook(m, grad_in, grad_out):
    ema_out[m] ← β·ema_out[m] + (1-β)·mean_batch(grad_out[0]²)  # gradient signal

for m in target_modules:
    fwd_handles[m] = m.register_forward_hook(fwd_hook)
    bwd_handles[m] = m.register_full_backward_hook(bwd_hook)

for t = 1 … T_stat:
    loss = model(batch).loss        # full-model forward (base model only, no adapter)
    loss.backward()                 # full-model backward; hooks fire on target modules
    optimizer.zero_grad()           # clear grads; NO optimizer.step()

for m in target_modules:
    fwd_handles[m].remove(); bwd_handles[m].remove()

# Stability check:
if any(Jaccard(top_{2K}(ema_in[m]), window=last_5_steps) < 0.85 for m):
    extend T_stat by 50 steps, re-run loop above (once)

─── PHASE 1: ALLOCATION ──────────────────────────────────────────────────────
for m in target_modules:
    slots_R[m] ← top2K_consecutive_pairs(ema_in[m])   # K disjoint Givens pairs
    slots_L[m] ← top2K_consecutive_pairs(ema_out[m])
    U[m]       ← sorted(dims(slots_R[m]) ∪ dims(slots_L[m]))  # |U| ≤ 4K=128
    δ[m]       ← nn.Parameter(zeros(|U[m]|))   # packed, shape [|U|]
    θ_R[m]     ← nn.Parameter(zeros(K))
    θ_L[m]     ← nn.Parameter(zeros(K))
    # U[m], slots_R[m], slots_L[m] stored as buffers (non-trainable)

─── PHASE 2: MAIN TRAINING (N optimizer steps) ───────────────────────────────
def jora_forward(m, x):
    # Right rotation: O(Kd) sparse; use temp buffer to avoid coordinate overwrite
    x̃ = x.clone()
    for k in range(K):
        i, j = slots_R[m][k]; c, s = cos(θ_R[m][k]), sin(θ_R[m][k])
        xi, xj = x̃[...,i].clone(), x̃[...,j].clone()   # temp buffers
        x̃[...,i], x̃[...,j] = c*xi - s*xj, s*xi + c*xj

    # D_sel via gather/scatter: O(|U|)
    x̂ = x̃.clone()
    x̂_U = x̃[..., U[m]]                          # gather, shape [..., |U|]
    x̂[..., U[m]] = (1 + δ[m]) * x̂_U             # scale by (1+δ)
    x̂[..., complement(U[m])] = 0.0               # zero outside U

    # Left rotation transposed: O(Kd) sparse
    ŷ = x̂.clone()
    for k in range(K-1, -1, -1):                  # reverse order = R_L^T
        i, j = slots_L[m][k]; c, s = cos(θ_L[m][k]), sin(θ_L[m][k])
        yi, yj = ŷ[...,i].clone(), ŷ[...,j].clone()
        ŷ[...,i], ŷ[...,j] = c*yi + s*yj, -s*yi + c*yj

    # Residualize: O(|U|)
    delta = ŷ.clone(); delta[..., complement(U[m])] = 0.0
    delta[..., U[m]] -= x[..., U[m]]              # subtract P_U·x

    return F.linear(x, m.weight, m.bias) + delta  # base + adapter

# Init check: θ=0, δ=0 → x̃=x, x̂[U]=x[U], x̂[∖U]=0, ŷ[U]=x[U], delta[U]=0 ✓

─── PHASE 3: MERGE FOR INFERENCE ─────────────────────────────────────────────
for m in target_modules:
    R_R  = build_givens_matrix(slots_R[m], θ_R[m], d)
    R_LT = build_givens_matrix_T(slots_L[m], θ_L[m], d)
    D    = zeros(d,d); D[U[m],U[m]] = diag(1 + δ[m])
    delta_W = R_LT @ D @ R_R
    delta_W[U[m], :] -= eye(d)[U[m], :]           # subtract P_U
    m.weight.data += delta_W                       # fuse; adapter removed
```

**Complexity summary**:

| Operation | Time | Extra memory |
|-----------|------|-------------|
| Calibration fwd+bwd (×T_stat) | Same as base model | O(d) EMA per module |
| R_sparse forward (per step) | O(Kd) ≈ 131K FLOPs/module | O(d) for clone |
| D_sel gather/scatter | O(\|U\|) ≤ 128 ops | — |
| Residualization | O(\|U\|) ≤ 128 ops | — |
| δ packed storage | — | \|U\|×4B ≤ 512B/module |
| AdamW optimizer states | — | 2×192×64 modules ≈ 0.1 MB total |
| Inference (post-merge) | Zero overhead | — |

Residualization cost: O(128) ops per module vs. O(16.8M) for W₀ matmul. Overhead < 0.001%.

### Core Mechanism

**Operator**: `D_sel = P_U + diag(δ)_U`, δ stored as packed vector, applied via gather/scatter.

**Zero-function-change**: θ=0, δ=0 → delta=0 ✓ (shown in Algorithm 1 init check).

**Gradient-live**: `∂delta/∂δ_u|_{init} = x_u ≠ 0` for u∈U; `∂delta/∂θ_R|_{init} ∝ P_U·∂R_R/∂θ·x ≠ 0` ✓

**Bilateral statistics** (honest, asymmetric):
- Left/output: gradient signal via backward hook: `ema_out[i] ← β·ema_out[i] + (1-β)·mean_batch((∂L/∂y_i)²)`
- Right/input: activation proxy via forward hook: `ema_in[j] ← β·ema_in[j] + (1-β)·mean_batch(x_j²)`
- Presented as "gradient-guided output allocation + activation-proxy input allocation." Not symmetric.

**Disjoint pairing**: stable-sort → top 2K → consecutive pairs → canonicalize.

**Precommitted interpretation rule** (claim-determining):
- Diag-only-selected (same U, R_R=R_L=I, only δ trained).
- Gap > 0.5pp on all three benchmarks → rotation claim stands.
- Gap ≤ 0.5pp → paper narrows to "adaptive sparse diagonal scaling with rotation-structured support selection." Still publishable.

### Training Plan

1. Mistral-7B-v0.1, Alpaca-cleaned 52k, SFT.
2. q_proj, o_proj: 64 modules total.
3. Calibration: T_stat=min(200,0.05·T_total) full-model fwd+bwd steps (hooks active, no optimizer step). ~5 min wall-clock. NOT in optimizer-step budget.
4. Stability check: Jaccard < 0.85 → extend 50 steps (once).
5. Allocation: δ=0 packed, θ=0, U=buffer (frozen).
6. Main training: N optimizer steps. AdamW, cosine decay, JORA-specific LR.
7. Variants: JORA-small (K=8, ~3K), JORA-base (K=32, ~12K).

### Failure Modes

- Jaccard < 0.85 after extension: report, increase β, flag limitation.
- δ explodes: L2 clip, detect via max|δ|.
- Diag-only-selected ≈ JORA: narrow claim per precommitted rule.
- qGOFT QA fail: retitle claim 2 to "JORA beats fixed-slot JORA," move qGOFT to appendix.
- JORA-base < LoRA-r1: report honest null-result Pareto story.

### Limitations

1. Square-layer restriction (q_proj, o_proj only; no rectangular layers).
2. Calibration wall-clock (~5 min per run).
3. Aggressive empirical bet: must-win (JORA-base > LoRA-r1) is the only directly validated Pareto claim.
4. Rotation contribution uncertain (Diag-only-selected precommitted rule applies).
5. Single dataset (Alpaca-cleaned) in main paper.

### Novelty and Elegance Argument

- **vs. qGOFT**: same rotation primitive, adaptive vs. static allocation.
- **vs. fixed-slot JORA**: same codepath, same budget, adaptive vs. random — clean isolation.
- **vs. LoRA**: rotation-structured adapter at 2–3% of LoRA-r1 parameter count.
- **Correctness (not novelty)**: residualized init is a necessary correctness property, not a headline contribution.
- **Focused**: one mechanism, one parameter regime, one clear Pareto story.

## Claim-Driven Validation Sketch

### Claim 1: JORA Pareto-dominates LoRA in extreme-budget regime

- JORA-small (~3K) and JORA-base (~12K) vs. LoRA-r1 (~524K), LoRA-r2 (~1M), LoRA-r4 (~2M).
- Metric: average MMLU / ARC-C / GSM8K vs. exact param count.
- Must-win: JORA-base > LoRA-r1; Primary: within 2pp of LoRA-r2 (if achieved); Stretch: ≥ LoRA-r2.
- Seeds: 3 for JORA-base and LoRA-r2; 2 for others.

### Claim 2 (Conditional): JORA beats qGOFT / static baseline at equal budget

**Conditional**: Claim 2 (JORA beats qGOFT) is made only if qGOFT passes QA: param count within 5% of target, training loss converges within 110% of LoRA-r1 loss. If QA fails, claim 2 is retitled to "JORA beats fixed-slot JORA at equal budget" and qGOFT moves to appendix.

- Experiment: JORA-base vs. faithful qGOFT (or fixed-slot JORA) at matched param count.
- Metric: average MMLU / ARC-C / GSM8K.
- Seeds: 3 each.

### Mechanism-Isolation (Mandatory)

- **Diag-only-selected** (claim-determining): same U, R_R=R_L=I, only δ trained.
- **Fixed-slot JORA**: same codepath, random slots. Tests allocation policy value.
- **Support stability curves**: Jaccard over calibration. Justifies one-shot allocation.
- **Non-residualized vs. residualized init** (appendix): d_sel=1 vs δ=0. Detection: loss spike > 0.1 nats in first 10 post-allocation steps.
- Seeds: 2 each.

### Appendix-Only

DoRA, OFT, BOFT breadth; JORA-large (K=64); calibration sensitivity (T_stat=100,200,400); non-residualized baseline.

## Experiment Handoff Inputs

- **Must-prove**: (1) JORA-base > LoRA-r1; (2) JORA-base > fixed-slot JORA or qGOFT (conditional).
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
- **Timeline**: Implementation 1d → baselines (qGOFT/fixed-slot) 1-2d → 1-seed screening 2d → multi-seed core 5d → analysis+tables 2d.
