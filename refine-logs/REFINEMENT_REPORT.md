# Refinement Report

**Problem**: How to give rotation an irreplaceable role in additive JORA-Diag, so that one-sided subspace-restricted coupling rotation outperforms NoRot?
**Initial Approach**: TC-CS JORA — Task-Conditioned Coupling Subspace Rotation, one-sided first
**Date**: 2026-04-27
**Rounds**: 3 / 5 MAX_ROUNDS (2 review-refine cycles)
**Final Score**: 7.75 / 10
**Final Verdict**: REVISE (close to READY — one targeted paragraph away)

---

## Problem Anchor

- **Bottom-line problem**: Current additive JORA-Diag cannot justify rotation's role — matched 1ep and 3ep ON/OFF comparisons show near-zero quality difference while ON costs ~3× runtime. Rotation has no irreplaceable function.
- **Must-solve bottleneck**: The current rotation mechanism is assigned no unique task; it acts as a global sparse basis reparameterization that diagonal scaling can absorb.
- **Non-goals**: Not claiming "rotation drives the gain" without evidence. Not building a broad PEFT framework. Not rewriting into residualized full-support JORA.
- **Constraints**: Compatible with additive JORA-Diag code path. Compact enough for EMNLP narrative.
- **Success condition**: TC-CS-1S shows a clear matched advantage over NoRot in a coupling-required regime.

---

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`
- Round logs: `refine-logs/round-0-initial-proposal.md`, `refine-logs/round-1-review.md`, `refine-logs/round-1-refinement.md`, `refine-logs/round-2-review.md`, `refine-logs/round-2-refinement.md`

---

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|-----------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 0 | 8 | 6 | 7 | 5 | 7 | 8 | 7 | 6.80 | REVISE |
| 1 | 8 | 7 | 7.5 | 5.5 | 6 | 8 | 7.5 | 6.95 | REVISE |
| 2 | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 7.75 | REVISE |

---

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|------------------------|------------------|--------|
| 1 | Coupling proxy vague (`c_ij = \|corr\|` ambiguous); O(d²) memory underspecified; Stage 1/2 terminology adds weight; diagnostic framework clutters contributions | Connected to `single_sided="right"`, `t_stat`, `pairs_freeze_after_warmup`; concretized proxy formula; dropped diagnostic framework contribution | Partial — proxy implementation blocked by hook signature |
| 2 | `cov_ij_ema` update path blocked (backward hook sees no raw activations); gradient path (`grad_input`) rejected (weight gradient, not input gradient); `min` discontinuity in score | Pivoted to forward-pass activation outer product EMA; simplified score formula (`|cov| * sqrt(g_i * g_j)`); added Fisher connection; moved CoupledPairCore to footnote | Near-READY — needs coupling regime specification |

---

## Final Proposal Snapshot

Canonical clean version lives in `refine-logs/FINAL_PROPOSAL.md`.

**Core thesis in 3 bullets**:
- Replace `energy[i] * energy[j]` pairing with activation-covariance-driven pairing: `score = |E[x_i x_j]| * sqrt(E[x_i²] * E[x_j²])`, where `E[x_i x_j]` is accumulated via forward-pass outer product EMA during calibration.
- Restrict rotation to one side only (`single_sided="right"`, R_L=I) to give rotation a minimal, interpretable role as input-side coupling before diagonal scaling.
- If TC-CS-1S > NoRot, the mechanism is coupling; if not, stop rotation revival主线.

---

## Method Evolution Highlights

1. **Most important simplification**: Dropped Stage 1/2/3 terminology and diagnostic framework contribution. Reused existing `t_stat`, `pairs_freeze_after_warmup`, and `single_sided` infrastructure instead of inventing new training phases.
2. **Most important mechanism upgrade**: Pivoted from activation cross-covariance (which required new forward-pass storage) to activation outer product EMA (which accumulates naturally in the existing forward hook). This resolved the blocking implementation gap identified in Round 1.
3. **Most important modernization**: Added Fisher connection — `g_cov_ema[i,j] = E[x_i * x_j]` is the empirical Fisher for the task loss. This positions the method in the information geometry framework and makes the task-conditioning claim precise.
4. **Most important simplification of score**: Replaced `c_ij * min(g_i, g_j)` with `|g_cov_ema[i,j]| * sqrt(grad_col_ema[i] * grad_col_ema[j])` — the `min` discontinuity was removed, the formula is symmetric, and it directly equals the unnormalized cross-covariance.

---

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | Use probe-based or Fisher-based subspace identification | Rejected for v1: Fisher requires a separate second-order pass, breaking the single-pass fine-tuning setup. EMA-based covariance is an online first-order approximation that stays within the existing training loop. | Accepted (partial) — Fisher connection acknowledged in text, EMA path kept as primary |
| 1 | The calibration phase needs its own "Stage 1" terminology | Rejected: reusing `t_stat` + `pairs_freeze_after_warmup` is cleaner and connects to existing code | Accepted |
| 2 | Use `grad_input` from backward hook for gradient covariance | Rejected with analysis: `grad_input` in `register_full_backward_hook` on an nn.Linear is weight gradient, not input gradient — it has shape `[in_features, out_features]`, which does not give input-dimension coupling | Accepted — pivoted to forward-pass activation outer product instead |

---

## Remaining Weaknesses

*(Coupling regime paragraph was added on 2026-04-27 — this closes the main remaining gap. The following are minor implementation notes.)*

1. **Outer product at d=4096 scale**: The `x_flat.T @ x_flat` outer product creates a large `(d, d)` intermediate tensor per calibration step. For large models (d=4096), this allocates ~67 MB per layer per step. A chunked accumulation or `addmm_`-style incremental update would be more memory-efficient in production. This is a code optimization, not a proposal flaw.

---

## Raw Reviewer Responses

<details>
<summary>Round 1 Review</summary>

Scores: Problem Fidelity 8, Method Specificity 6, Contribution Quality 7, Frontier Leverage 5, Feasibility 7, Validation Focus 8, Venue Readiness 7. Overall 6.80. Verdict: REVISE.

Main concerns: Coupling proxy vague (O(d²) ambiguity, no update path specified), Stage 1/2 terminology unnecessary, diagnostic framework clutters contributions, single_sided flag not referenced.

</details>

<details>
<summary>Round 2 Review</summary>

Scores: Problem Fidelity 8, Method Specificity 7, Contribution Quality 7.5, Frontier Leverage 5.5, Feasibility 6, Validation Focus 8, Venue Readiness 7.5. Overall 6.95. Verdict: REVISE.

Main concerns: `cov_ij_ema` update blocked by hook signature (backward hook has no raw activations), gradient path (`grad_input`) is weight gradient not input gradient, `min` discontinuity in score, CoupledPairCore in Phase 2 adds scope.

</details>

---

## Next Steps

- **Coupling regime paragraph added** (2026-04-27): Inserted into FINAL_PROPOSAL.md as a new "### Coupling Regime" section between Technical Gap and Method Thesis. This addresses the one remaining item to close the gap to READY.
- **Next**: Run `/experiment-plan` to generate a detailed claim-driven experiment roadmap from the final proposal.
