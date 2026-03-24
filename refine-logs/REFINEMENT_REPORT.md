# Refinement Report

**Problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
**Initial Approach**: Vague rotation-based PEFT adapter combining bilateral statistics, adaptive slot allocation, and Givens rotations.
**Date**: 2026-03-18
**Rounds**: 9 / 9
**Final Score**: 8 / 10
**Final Verdict**: REVISE (empirical hedge — not a method critique)

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope) beats LoRA-r1 as a must-win; within 2pp of LoRA-r2 (primary if achieved); matches LoRA-r2 (stretch if achieved). Beats qGOFT at equal budget (conditional on reproduction).

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 6                | 5                  | 6                    | 8                 | 6           | 6                | 6               | 6       | REVISE  |
| 2     | 8                | 6                  | 7                    | 8                 | 6           | 8                | 7               | 7       | REVISE  |
| 3     | 8                | 7                  | 8                    | 8                 | 7           | 8                | 7               | 8       | REVISE  |
| 4     | 8                | 8                  | 7                    | 8                 | 8           | 6                | 7               | 8       | REVISE  |
| 5     | 8                | 8                  | 7                    | 8                 | 6           | 8                | 7               | 8       | REVISE  |
| 6     | 8                | 6                  | 8                    | 8                 | 5           | 9                | 6               | 7       | REVISE  |
| 7     | 8                | 7                  | 8                    | 8                 | 7           | 9                | 7               | 8       | REVISE  |
| 8     | 8                | 8                  | 8                    | 8                 | 8           | 9                | 7               | 8       | REVISE  |
| 9     | 8                | 9                  | 8                    | 8                 | 8           | 9                | 8               | 8       | REVISE  |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1     | Vague mechanism; no pseudocode; bilateral story hand-wavy | Introduced core bilateral rotation operator, calibration phase, zero-init proof | Partial |
| 2     | Algorithm unpseudocoded; no forward pass detail | Wrote concrete forward pass, zero-function-change init proof | Partial |
| 3     | No disjoint pairing rule; EMA formula underspecified | Stable-sort consecutive-pair rule; explicit EMA accumulation formula | Resolved |
| 4     | Experiment sprawl; validation too broad | Capped to 3 core experiment blocks; breadth baselines moved to appendix | Resolved |
| 5     | Contribution sprawl; δ full-width waste | One dominant contribution; packed δ storage introduced | Resolved |
| 6     | R_sparse in-place coordinate overwrite bug risk | Added temp buffer clone before Givens loop | Resolved |
| 7     | Calibration `torch.no_grad()` inconsistent with gradient collection | Rewrote calibration as full-model fwd+bwd with hooks, no no_grad | Resolved |
| 8     | δ storage form ambiguous; claim 2 conditionality informal | Specified packed vector + gather/scatter; explicit qGOFT QA condition | Resolved |
| 9     | R_sparse temp buffer safety; qGOFT QA in paper text | All four action items addressed; Algorithm 1 fully implementable | Resolved (minor) |

## Final Proposal Snapshot

Canonical clean version: `refine-logs/FINAL_PROPOSAL.md`

**Core thesis**: JORA allocates one fixed set of sparse bilateral rotation slots using early task statistics (gradient-guided output, activation-proxy input), then trains Givens angles and a packed diagonal correction on the frozen support. At 12,288 total parameters (2.34% of LoRA-r1), it claims Pareto-dominance over LoRA in the extreme-budget regime on square attention layers.

Key bullets:
- **Operator**: `D_sel = P_U + diag(δ)_U`, applied via gather/scatter on support U.
- **Forward**: right Givens rotation → D_sel → left Givens transposed → residualize → add to base output.
- **Zero-init**: θ=0, δ=0 → delta=0; all parameters gradient-live from calibration end.
- **Merge**: `W_merged = W₀ + R_L^T · D_sel · R_R − P_U` — exact, zero inference overhead.
- **Claim-determining ablation**: Diag-only-selected (same U, no rotations). If within 0.5pp, rotation claim is narrowed.

## Method Evolution Highlights

1. **Most important simplification**: Removed full-width D_diag and all tempting-but-excluded components (repeated reseating, rectangular layers, tanh merge, OER). The method went from a vague multi-component system to a single operator with 192 params/module.
2. **Most important mechanism upgrade**: Calibration block rewritten from internally inconsistent `torch.no_grad()` + `autograd` to a clean full-model fwd+bwd with `register_forward_hook` / `register_full_backward_hook` — now literally implementable.
3. **Most important modernization decision**: Intentionally did NOT add LLM/VLM/RL-era components. Reviewer confirmed this is appropriate. Rotation-based PEFT is already a focused, principled primitive.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 3     | OFT/BOFT should be baselines | Accepted as appendix-only; too expensive for main budget | Partial accept |
| 5     | Consider right-side gradient-only (full bilateral) | Accepted asymmetric framing as honest; activation proxy kept for input side | Accepted framing |
| 6     | Add DoRA/OFT/BOFT breadth | Kept appendix-only; main story is Pareto not breadth | Partial reject |
| 7–9   | No drift detected | N/A | N/A |

## Remaining Weaknesses

1. **Empirical bet unvalidated**: JORA-base beating LoRA-r1 at 2.34% of its parameter count is a strong claim. The 8/10 score plateau is because the top-venue case depends on this empirical outcome.
2. **Asymmetric bilateral statistics**: Right-side (input) uses activation proxy, not gradient signal. The bilateral story is factually asymmetric; this is accepted but limits the theoretical elegance argument.
3. **Diag-only-selected uncertainty**: The precommitted narrowing rule is honest, but if δ-only matches full JORA, the rotation contribution becomes a support-selection mechanism only.
4. **Single dataset**: Alpaca-cleaned only in main paper. Cross-dataset generalization deferred to appendix or future work.

## Raw Reviewer Responses

<details>
<summary>Round 1 Review</summary>

See: `refine-logs/round-1-review.md`

</details>

<details>
<summary>Round 2 Review</summary>

See: `refine-logs/round-2-review.md`

</details>

<details>
<summary>Round 3 Review</summary>

See: `refine-logs/round-3-review.md`

</details>

<details>
<summary>Round 4 Review</summary>

See: `refine-logs/round-4-review.md`

</details>

<details>
<summary>Round 5 Review</summary>

See: `refine-logs/round-5-review.md`

</details>

<details>
<summary>Round 6 Review</summary>

See: `refine-logs/round-6-review.md`

</details>

<details>
<summary>Round 7 Review</summary>

See: `refine-logs/round-7-review.md`

</details>

<details>
<summary>Round 8 Review</summary>

See: `refine-logs/round-8-review.md`

</details>

<details>
<summary>Round 9 Review</summary>

See: `refine-logs/round-9-review.md`

</details>

## Next Steps

- **Status**: REVISE — not because design remains underspecified, but because the top-venue case depends on the empirical outcome.
- **Immediately actionable**: proceed to `/experiment-plan` for a detailed claim-driven experiment roadmap, then begin implementation and baseline runs.
- **Priority experiments**:
  1. JORA-base (K=32, ~12K params) vs. LoRA-r1 (~524K params) — must-win
  2. Diag-only-selected vs. JORA-base — claim-determining ablation
  3. Fixed-slot JORA vs. JORA-base — allocation policy validation
  4. qGOFT reproduction QA — conditional claim 2
