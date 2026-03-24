# JORA Research Review — GPT-5.4 xhigh (NeurIPS/ICML Level)
Date: 2026-03-18
Codex threadId: `019cfebc-06bf-7d93-a632-2f85b6a63363`
Verdict: Hard reject → Borderline with fixes

---

## Round 1: Initial Review (9-Point Hard Reject)

### Core Issues Identified

1. **Code-paper mismatch (paper-breaking / gatekeeping)**
   Paper states: `y = W₀x + M ⊙ tanh(R_L^T · D · R_R · x)`
   Code does: `out = scale * (base_out + delta)` — magnitude scales the **full output**, not just delta.
   Merge path also inconsistent: applies magnitude only to `delta_weight`.
   → **Fix required before any review can be positive. This is #1 priority.**

2. **Catastrophic empirical result**
   MMLU: JORA=0.2448 vs LoRA-r4=0.4361 vs pretrained=0.4115
   → Identified as a broken run (LR bug fixed in commit `62e5371`). Remove from experiment tree.

3. **qGOFT prior art (ICML 2024)**
   Ma et al. already use Givens rotations for orthogonal fine-tuning. Directly weakens novelty claim.
   → Acknowledge as primary prior, include in main table, show JORA adds bilateral sparse rotation + EMA selection + OER on top.

4. Weak theory (overclaims)
5. Non-mergeable due to tanh (deployment concern)
6. Unfair parameter efficiency comparison
7. Poor deployment story
8. Limited to one model family
9. Stale failed-result files visible in experiment tree

---

## Round 2: Responses and Q&A

### Code-Paper Fix Decision
**Decision: Option A (fix code, not math)**
Change forward pass to: `out = base_out + scale * delta`
This is the "delta-only OER" variant. Run full-output OER as an ablation only.
If full-output OER wins ablation + gain-only controls, can promote in future work.

The corrected paper equation (with tanh retained): `y = s ⊙ W₀x + s ⊙ tanh(R_L^T D R_R x)` — still messy.
Cleaner: drop tanh entirely → `y = W₀x + s ⊙ (R_L^T D R_R x)` → fully mergeable.
**Recommend: drop tanh, adopt delta-only OER, get a mergeable method.**

Merged weight form: `W_merged = W₀ + Diag(s) · R_L^T · D · R_R` (for diagonal core, approximately)

### qGOFT Baseline Strategy
- **Best**: Timebox 1-2 days to reimplement qGOFT (simple: Givens rotation on weight matrix rows), label as "our reimplementation"
- **Fallback**: Use BOFT + HRA or RoAd as orthogonal baselines, cite qGOFT as "not reproduced in our setup"
- **BOFT alone is insufficient** — need at least one more orthogonal baseline if qGOFT dropped

### Empirical Story: Efficiency-Pareto (Recommended)
Instead of "beats LoRA-r4 at equal budget" (weak), tell the **Pareto story**:
- JORA at half LoRA-r4 parameter budget **matches** LoRA-r4
- JORA at half budget **clearly beats** LoRA-r2
- JORA sits on a **better accuracy-vs-params frontier**
→ Report actual trainable parameter counts, not "r-equivalent"

---

## Mock NeurIPS 2026 Review (Post-Fix, Optimistic Scenario)

**Summary**: JORA — sparse bilateral Givens rotations + dynamic EMA selection + OER calibration. Extends qGOFT/OFT line. Shows modest but consistent gains over LoRA-r4 at matched budget and over qGOFT.

**Score: 6/10** (borderline accept / weak accept)
Drops to **5/10** if gains over LoRA-r4 are within seed variance.

**Strengths**:
- Targets meaningful gap: adaptive vs fixed rotation structure
- Closest prior acknowledged and included empirically
- Better-than-average ablation package
- Parameter-budget aware comparison

**Weaknesses**:
- Moderate novelty (extension of qGOFT/OFTv2/HRA line)
- Theoretical claims must be softened substantially
- Single model family
- Must rule out "row-gating without orthogonality" as driver

**What Would Move to Accept**:
- Second backbone or stronger training corpus
- Clean efficiency story (VRAM + throughput)
- Gain-only control (if full-output OER used)
- Trimmed theory matching actual claims

---

## Acceptance-Lift Priority Ranking

### Gatekeeping (must-do before submission):
1. **Fix code-paper mismatch** — paper is dead without this

### Score lift (after method is valid):
1. **Clean empirical results with Pareto story** — highest impact
2. **Closest orthogonal baseline (qGOFT or BOFT+HRA/RoAd)** — beat it
3. **Gain-only control** — only if promoting full-output OER

**One sentence**: highest-impact path = valid implementation + better Pareto frontier than LoRA + one credible orthogonal baseline.

---

## Minimum 2-Week Experiment Package (3× RTX 4090)

| Priority | Task | Notes |
|----------|------|-------|
| 1 | Fix code-paper mismatch | Drop tanh, delta-only OER, make mergeable |
| 2 | Remove stale failed experiment JSONs | Clean experiment tree |
| 3 | Main model: Mistral-7B | Preferred; Llama-2-7B if unstable |
| 4 | Training: one SFT dataset, Alpaca-cleaned or 50k subset | Speed over diversity |
| 5 | Eval: MMLU 5-shot, ARC-C, GSM8K | Fixed protocol across all methods |
| 6 | Baselines: LoRA-r2, LoRA-r4, DoRA-r4, qGOFT (or BOFT+HRA) | 2 seeds for non-JORA |
| 7 | JORA variants: full, no-selection, no-OER | 3 seeds for JORA-full + LoRA-r4 |
| 8 | Budget: JORA at ~half LoRA-r4 params, also equal-budget | Pareto story |
| 9 | Efficiency: VRAM + tokens/sec | All methods |

---

## Theory: OER Conservation Proposition
The OER conservation fact (sum of squared row norms preserved at init) is worth a short Proposition/Lemma:
- Frame as: "OER induces a fixed row-energy budget in scale space by construction"
- Do **not** call it "output energy conservation" (misleading for delta-only variant)
- Place in methods or appendix, NOT as a headline theorem

---

## Key References
- qGOFT (ICML 2024): https://proceedings.mlr.press/v235/ma24a.html
- OFT (NeurIPS 2023): https://proceedings.neurips.cc/paper_files/paper/2023/file/faacb7a4827b4d51e201666b93ab5fa7-Paper-Conference.pdf
- BOFT (ICLR 2024): https://openreview.net/forum?id=7NzgkEdGyr
- OFTv2 (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.1627/
- HRA (2024): https://arxiv.org/abs/2405.17484
- RoAd (2024): https://arxiv.org/abs/2409.00119
- DoRA (2024): https://arxiv.org/abs/2402.09353
- SHiRA (2024): https://arxiv.org/abs/2406.13175

---

## Immediate Action Items (Ordered by Priority)

- [ ] **FIX: code-paper mismatch** — change forward pass to `out = base_out + scale * delta`; optionally drop tanh for mergeability
- [ ] **FIX: merge path** — align `_apply_magnitude_to_delta_weight` with new forward convention
- [ ] **CLEAN: experiment tree** — remove/quarantine `experiment/llama2_7b/jora/s16_k4_diag/` broken run
- [ ] **IMPLEMENT: qGOFT baseline** (1-2 day timebox); fallback to BOFT+HRA
- [ ] **RUN: main experiments** — Mistral-7B, Alpaca-cleaned, MMLU/ARC-C/GSM8K, 3 seeds
- [ ] **STORY: Pareto frontier** — JORA half-budget vs LoRA-r2/r4
- [ ] **ABLATION: delta-only OER vs full-output OER vs no-OER vs gain-only**
- [ ] **UPDATE: paper math** — rewrite forward equation to match fixed code
- [ ] **TRIM: theory section** — remove overclaims, keep only the conservation Proposition
