# Experiment Plan

**Problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
**Method Thesis**: JORA allocates one fixed set of sparse bilateral rotation slots using early task statistics, then trains Givens angles + packed diagonal correction — achieving Pareto-dominant accuracy at extreme parameter budgets (3K–25K vs LoRA-r1's 524K).
**Date**: 2026-03-18

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1 (must-win) | Core Pareto story: JORA achieves useful accuracy at 2.3% of LoRA-r1 params | JORA-base (12K) > LoRA-r1 (524K) on avg MMLU/ARC-C/GSM8K | B1, B2 |
| C1 (primary) | JORA is competitive at extreme budget | JORA-base within 2pp of LoRA-r2 (1M params) | B1 |
| C2 (conditional) | Adaptive allocation beats static allocation | JORA-base > fixed-slot JORA at equal budget; OR JORA-base > qGOFT if qGOFT passes QA | B3 |
| C3 (mechanism-isolation) | Rotation contribution is real, not just support selection | JORA-base > diag-only-selected (same U, no rotations) by >0.5pp | B4 |

---

## Paper Storyline

**Main paper must prove**:
- JORA-base beats LoRA-r1 at 2.3% of its parameter count (must-win, Table 1)
- JORA-base meaningfully closes gap to LoRA-r2 (primary headline, Table 1)
- Adaptive allocation adds value over random slots (B3 ablation, Table 2)
- Rotation angles contribute beyond support selection (B4 claim-determining ablation, Table 2)

**Appendix can support**:
- JORA-large (K=64) scaling behavior
- Calibration sensitivity (T_stat=100/200/400)
- Non-residualized vs residualized init (loss spike detection)
- qGOFT breadth comparison (if QA passes)
- DoRA/OFT/BOFT breadth (optional)
- Support stability Jaccard curves

**Experiments intentionally cut**:
- k/v/MLP scope extension (non-square layers)
- Repeated reseating schedule
- JORA with tanh merge path (already excluded from proposal)
- Full-width diagonal core comparison

---

## Experiment Blocks

### Block B0: Sanity and Pipeline Validation
- **Claim tested**: N/A (infrastructure gate)
- **Why this block exists**: Ensure training converges, MMLU scoring is correct, parameter counts match theory, adapter save/load is clean.
- **Dataset / split / task**: yahma/alpaca-cleaned, train split; MMLU (cais/mmlu, all, test)
- **Compared systems**: JORA-base (K=32) on OPT-350M, LoRA-r1 on OPT-350M, quick overfit check
- **Metrics**: Training loss decreasing (sanity), MMLU accuracy, exact trainable param count
- **Setup details**: OPT-350M (small, fast), 500 steps, 1 seed, GPU=1 RTX 4090. Use `export HF_ENDPOINT=https://hf-mirror.com` for dataset mirror.
- **Success criterion**: Loss decreases; JORA param count = 192 × (num_target_modules); MMLU eval runs without error; adapter checkpoint save/load works.
- **Failure interpretation**: If MMLU scoring is broken → fix before proceeding. If param count wrong → fix JoraConfig. If loss explodes → check LR, theta_init_std.
- **Table / figure target**: Not in paper; internal gate only.
- **Priority**: MUST-RUN (first)

---

### Block B1: Main Anchor — Pareto Frontier
- **Claim tested**: C1 (must-win + primary)
- **Why this block exists**: Core paper claim. Without this, there is no paper.
- **Dataset / split / task**: yahma/alpaca-cleaned (52K), train; eval on MMLU (full test) + ARC-Challenge + GSM8K
- **Compared systems**:
  - JORA-small (K=8, ~3K params)
  - JORA-base (K=32, ~12K params)
  - LoRA-r1 (~524K params, q+o scope)
  - LoRA-r2 (~1M params, q+o scope) — primary comparison
  - LoRA-r4 (~2M params, q+o scope) — upper anchor
- **Metrics**: Average accuracy on MMLU / ARC-C / GSM8K (primary); per-task breakdown (secondary); exact trainable param count reported for all runs
- **Setup details**:
  - Backbone: Mistral-7B-v0.1 (preferred) or Llama-2-7B if Mistral unavailable
  - target_modules: q_proj, o_proj (square layers only, 32 layers × 2 = 64 modules)
  - Seeds: 3 for JORA-base and LoRA-r2 (key comparison); 2 for JORA-small, LoRA-r1, LoRA-r4
  - Optimizer: AdamW, cosine LR decay, warmup_ratio=0.03
  - JORA-specific LR: separate lr_theta and lr_core (sweep B0 first); LoRA LR: standard 2e-4
  - Steps: 2000–5000 (match LoRA-r2 convergence budget)
  - batch_size: 8 per device, gradient_accumulation=2 on 3× RTX 4090
  - JORA config: selection=topk_ema, ema_beta=0.99, T_stat=min(200, 0.05*total_steps), magnitude=none (OER excluded from core; ablated in B4)
  - LoRA config: alpha=2r, dropout=0.0
  - Use HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`
- **Success criterion (must-win)**: JORA-base avg > LoRA-r1 avg on all three benchmarks, mean ± std shown
- **Success criterion (primary)**: JORA-base avg within 2pp of LoRA-r2 avg
- **Failure interpretation**: If JORA-base < LoRA-r1 → honest null-result Pareto story; paper claims "rotation structure does not degrade vs LoRA at extreme budget" and pivots to ablation story. Must still run B3+B4.
- **Table / figure target**: Table 1 (main results), Figure 1 (Pareto frontier scatter plot: accuracy vs log param count)
- **Priority**: MUST-RUN

---

### Block B2: Hyperparameter Screening (LR + Budget)
- **Claim tested**: Supporting C1 (ensures fair comparison)
- **Why this block exists**: JORA-specific learning rates (lr_theta, lr_core) are not well-established. A 1-seed sweep on a small model ensures the main B1 run uses fair hyperparameters.
- **Dataset / split / task**: yahma/alpaca-cleaned, train; MMLU (200-sample subset for speed)
- **Compared systems**: JORA-base with lr_theta ∈ {1e-3, 5e-3, 1e-2, 5e-2} × lr_core ∈ {5e-4, 1e-3, 5e-3}
- **Metrics**: MMLU accuracy (200 samples), training loss at 500 steps
- **Setup details**: OPT-350M or Mistral-7B (1 GPU), 500 steps, 1 seed. 12-run grid.
- **Success criterion**: Identify best (lr_theta, lr_core) pair; verify it does not cause loss explosion.
- **Failure interpretation**: If all LRs give similar results → use default from prior work; if all diverge → investigate theta_init_std.
- **Table / figure target**: Appendix (LR sensitivity)
- **Priority**: MUST-RUN (run before B1)

---

### Block B3: Allocation Policy Validation — Fixed-Slot vs Adaptive
- **Claim tested**: C2 — adaptive allocation adds value over static random slots
- **Why this block exists**: Core novelty of JORA is the EMA-guided slot allocation. If random slots match adaptive slots, the contribution is just "sparse rotation" not "adaptive sparse rotation."
- **Dataset / split / task**: Same as B1 (Mistral-7B, alpaca-cleaned, MMLU/ARC-C/GSM8K)
- **Compared systems**:
  - JORA-base (adaptive, topk_ema) — already run in B1
  - Fixed-slot JORA (same codepath, selection=random, same K=32, same budget)
  - (Conditional) qGOFT at matched parameter budget — if QA passes
- **qGOFT QA condition**: qGOFT is included if: (a) faithful reimplementation runs in PEFT framework, (b) param count within 5% of 12K, (c) training loss converges within 110% of LoRA-r1's final loss. If QA fails, claim 2 becomes "JORA beats fixed-slot JORA."
- **Metrics**: Same as B1 (avg MMLU/ARC-C/GSM8K); also report Jaccard stability of top-2K indices over calibration
- **Setup details**: Seeds: 3 each. Same training config as B1 for fair comparison.
- **Success criterion**: JORA-base > fixed-slot JORA by statistically meaningful margin (>0.5pp avg, p<0.1 over 3 seeds)
- **Failure interpretation**: If fixed-slot ≈ JORA-base → contribution narrows to "sparse rotation at extreme budget," still publishable as ablation story. Report honestly.
- **Table / figure target**: Table 2 (ablation), Figure 2 (Jaccard stability curves)
- **Priority**: MUST-RUN

---

### Block B4: Mechanism Isolation — Diag-Only-Selected (Claim-Determining)
- **Claim tested**: C3 — Givens rotation angles contribute meaningfully beyond support selection
- **Why this block exists**: Precommitted claim-determining ablation. If diag-only-selected (same U, R_R=R_L=identity, only δ trained) matches JORA-base within 0.5pp, the rotation contribution is downgraded to "support selection mechanism only." Paper narrows claim proactively.
- **Dataset / split / task**: Same as B1
- **Compared systems**:
  - JORA-base (K=32, θ+δ trained) — already run in B1
  - Diag-only-selected (same U from B1 allocation, θ_L=θ_R=frozen at 0, only δ trained; S_L=0, S_R=0 in JoraConfig but U preserved)
- **Metrics**: Avg MMLU/ARC-C/GSM8K
- **Setup details**: Seeds: 2 each. Same U (save and reuse slot allocation from B1 JORA-base run). Only δ is trainable; θ params frozen or removed.
- **Success criterion**: JORA-base > diag-only-selected by >0.5pp avg on all three benchmarks
- **Failure interpretation**: Gap ≤ 0.5pp → paper narrows claim to "adaptive sparse diagonal scaling with rotation-structured support selection." Still publishable. Rotation result moves to appendix.
- **Table / figure target**: Table 2 (ablation row)
- **Priority**: MUST-RUN

---

### Block B5: Support Stability Diagnostics (Jaccard Curves)
- **Claim tested**: Justifies one-shot allocation (supports C2)
- **Why this block exists**: One-shot static allocation is only justified if top-2K indices converge during calibration. Jaccard curves over calibration steps are the diagnostic.
- **Dataset / split / task**: Subset (first 200 calibration steps) from B1 JORA-base run
- **Compared systems**: N/A — diagnostic only
- **Metrics**: Jaccard(top_{2K}(ema_in), t, t-1) over T_stat steps, per layer and averaged
- **Setup details**: Log ema_in snapshots every 20 calibration steps during B1 training. Compute Jaccard post-hoc.
- **Success criterion**: Jaccard stabilizes > 0.85 within T_stat steps
- **Failure interpretation**: If Jaccard < 0.85 → calibration extension fires; report β=0.99 stability, flag in limitations.
- **Table / figure target**: Figure 2 (Jaccard curves, appendix or main depending on claim 2 strength)
- **Priority**: MUST-RUN (piggybacks on B1 logging)

---

### Block B6 (Appendix): Calibration Sensitivity
- **Claim tested**: Robustness of allocation policy
- **Why this block exists**: Show that T_stat=200 is stable enough; results not highly sensitive to calibration budget.
- **Dataset / split / task**: Same as B1
- **Compared systems**: JORA-base with T_stat ∈ {50, 100, 200, 400}
- **Metrics**: Avg MMLU/ARC-C/GSM8K
- **Setup details**: 1 seed each, Mistral-7B. 4 runs.
- **Success criterion**: Performance plateau by T_stat=200, confirming our default.
- **Failure interpretation**: High sensitivity → increase T_stat for main results, flag in limitations.
- **Table / figure target**: Appendix Table A1
- **Priority**: NICE-TO-HAVE

---

### Block B7 (Appendix): Non-Residualized Init Baseline
- **Claim tested**: Zero-function-change init is necessary
- **Why this block exists**: Shows that D_sel=1 (non-residualized) causes training instability (loss spike > 0.1 nats in first 10 post-allocation steps).
- **Dataset / split / task**: Same as B1 (1 seed, 500 steps)
- **Compared systems**: JORA-base residualized (δ=0) vs non-residualized (D_sel starts as identity block on U)
- **Metrics**: Training loss at steps 1–10 post-allocation; final MMLU
- **Setup details**: 1 seed. Modify init to D_sel=I_U to create non-residualized variant.
- **Success criterion**: Loss spike > 0.1 nats in non-residualized; stable in residualized.
- **Failure interpretation**: No spike → residualized init is still correct but the "necessity" argument is weakened. Move to footnote.
- **Table / figure target**: Appendix Figure A1
- **Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost (GPU-hrs) | Risk |
|-----------|------|------|---------------|----------------|------|
| M0: Sanity | Pipeline works end-to-end | B0 (OPT-350M, 1 seed, 500 steps) | Loss decreasing + correct param count + MMLU runs | ~2h | Low |
| M1: LR Sweep | Identify fair JORA LR | B2 (12-run grid, OPT-350M) | Best (lr_theta, lr_core) found | ~6h | Low |
| M2: 1-Seed Screening | Directional signal before multi-seed | B1 (1 seed, Mistral-7B, 2000 steps) + B3 fixed-slot 1 seed + B4 diag-only 1 seed | JORA-base > LoRA-r1 directionally | ~20h | Medium |
| M3: Multi-Seed Core | Paper-quality evidence | B1 (3 seeds JORA-base + LoRA-r2; 2 seeds JORA-small, LoRA-r1, LoRA-r4) + B3 (3 seeds) + B4 (2 seeds) | Must-win confirmed with error bars | ~90h | High (core empirical bet) |
| M4: Diagnostics | Jaccard curves + qGOFT QA | B5 (Jaccard from B1 logs) + qGOFT QA check | Jaccard > 0.85; qGOFT pass/fail decision | ~10h | Medium |
| M5: Polish | Appendix + figures | B6 + B7 | All appendix runs complete | ~12h | Low |

**Total estimated GPU-hours**: ~140h (3× RTX 4090, ~2 weeks)

---

## Compute and Data Budget

- **Total estimated GPU-hours**: 140h (conservative estimate including reruns)
- **Data preparation**: yahma/alpaca-cleaned (auto-download), MMLU/ARC-C/GSM8K (auto via datasets); use `export HF_ENDPOINT=https://hf-mirror.com`
- **Human evaluation**: None needed
- **Biggest bottleneck**: M3 multi-seed on Mistral-7B (~90h). If budget is tight, reduce seeds (min 2 for main comparison) and limit LoRA-r4 to 1 seed.
- **Secondary bottleneck**: qGOFT reimplementation (1-2 day timebox; fallback = fixed-slot JORA only)

---

## Risks and Mitigations

- **Risk**: JORA-base < LoRA-r1 (core empirical bet fails)
  **Mitigation**: Report honest null-result Pareto story; shift paper to "rotation structure at extreme budget" comparison (JORA vs diag-only, JORA vs fixed-slot); ablation story may still be publishable at workshop venues. Check LR and T_stat sensitivity first.

- **Risk**: qGOFT reimplementation fails QA
  **Mitigation**: Retitle Claim 2 to "JORA beats fixed-slot JORA" (already precommitted). Move qGOFT to appendix.

- **Risk**: Diag-only-selected ≈ JORA-base (rotation claim uncertain)
  **Mitigation**: Precommitted narrowing: paper says "adaptive sparse diagonal scaling with rotation-structured support selection." Still publishable; rotation contribution downgraded but not eliminated.

- **Risk**: OER/magnitude module mismatch between code and paper
  **Mitigation**: Per GPT-5.4 review: fix forward pass to `out = base_out + scale * delta` (delta-only OER). Exclude OER from main B1 runs (magnitude=none); test OER in appendix only. This avoids gatekeeping issue.

- **Risk**: Broken training run pollutes results
  **Mitigation**: Per GPT-5.4 review: quarantine any run with MMLU < pretrained baseline (e.g., OPT-350M pretrained ~0.26; Mistral-7B pretrained ~0.60). Flag and re-run.

- **Risk**: Memory pressure on 3× RTX 4090 with Mistral-7B
  **Mitigation**: Use bfloat16, gradient checkpointing, batch_size=4 with accumulation=4. Or use OPT-1.3B as fallback backbone (report honestly).

---

## Final Checklist

- [ ] Main paper tables are covered (B1 → Table 1, B3+B4 → Table 2)
- [ ] Novelty is isolated (B3: adaptive vs fixed-slot)
- [ ] Simplicity is defended (single mechanism; OER excluded from main)
- [ ] Frontier contribution is justified (rotation primitive — intentionally conservative; rotation necessity tested in B4)
- [ ] Nice-to-have runs (B6, B7) are separated from must-run runs
- [ ] Code-paper mismatch is fixed before running B1 (OER = delta-only; merge path consistent)
- [ ] Broken run (if any) is quarantined before reporting
- [ ] qGOFT QA pass/fail decision is documented before claim 2 is written
- [ ] Param counts verified empirically for all adapters (not just estimated)
