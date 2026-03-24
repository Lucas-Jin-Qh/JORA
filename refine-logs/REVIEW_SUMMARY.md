# Review Summary

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

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|-------------------------|------------------------------------------|---------|----------------|
| 1     | No concrete method; vague bilateral story; low specificity | Introduced core bilateral rotation operator, Givens pairs, calibration phase | Partial | Mechanism still underspecified |
| 2     | Algorithm still unpseudocoded; forward pass not concrete; no zero-init proof | Wrote concrete forward pass pseudocode; added zero-function-change init proof | Partial | Calibration ambiguous |
| 3     | No disjoint pairing rule; EMA accumulation underspecified | Added stable-sort consecutive-pair rule; EMA formula made explicit | Yes | Claim 2 not conditionalized |
| 4     | Validation too broad; experiment sprawl | Capped to 3 core experiment blocks; moved breadth baselines to appendix | Yes | qGOFT conditionality informal |
| 5     | Contribution sprawl; feasibility concerns; δ full-width waste | Narrowed to one dominant contribution; packed δ storage introduced | Yes | Calibration pseudocode still inconsistent |
| 6     | R_sparse in-place bug risk; forward pass unsafe | Added `x.clone()` temp buffer before Givens loop; marked in-place issue | Yes | Calibration block still internally inconsistent |
| 7     | Calibration `torch.no_grad()` vs gradient extraction mismatch | Rewrote calibration as full-model fwd+bwd with layer hooks, no no_grad wrapper | Yes | δ storage form still unspecified |
| 8     | δ implementation form ambiguous; claim 2 conditionality informal | Specified packed δ vector + gather/scatter; formalized claim 2 QA condition | Yes | Action items minor |
| 9     | R_sparse temp buffer (minor); qGOFT QA in paper text | All four action items addressed; Algorithm 1 now fully implementable | Yes | Score plateau at 8; empirical outcome unknown |

## Overall Evolution

- **Concreteness**: Moved from a vague "adaptive rotation adapter" description to a fully specified Algorithm 1 with pseudocode covering all four phases (calibration hooks, allocation, forward pass, merge). An engineer can implement it directly.
- **Contribution focus**: Narrowed from a multi-claim system to one dominant mechanism: one-shot adaptive allocation of sparse bilateral rotation slots inside rotation-based PEFT.
- **Unnecessary complexity removed**: Full-width D_diag, repeated reseating, rectangular operators (k/v, MLP), tanh merge, OER, and DoRA/OFT/BOFT breadth all moved to appendix or excluded entirely.
- **Modern leverage**: Stayed intentionally conservative — rotation-based PEFT is a mature primitive. No forced LLM/VLM/RL components. Reviewer confirmed this is appropriate ("Nothing feels old-school or trend-forced").
- **Drift**: Never occurred. All 9 anchor checks confirmed the method still targets the original bottleneck.

## Final Status

- **Anchor status**: Preserved across all 9 rounds.
- **Focus status**: Tight. One mechanism, one parameter regime, one Pareto story.
- **Modernity status**: Intentionally conservative — appropriate for this problem.
- **Strongest parts of final method**:
  1. Algorithm 1 is fully implementable: calibration hooks → allocation → packed δ forward → merge path.
  2. Zero-function-change initialization is proven and verified in pseudocode.
  3. Precommitted Diag-only-selected rule makes the rotation contribution falsifiable and claim-determining.
  4. Claim 2 conditionality on qGOFT QA is clean and honest.
  5. Parameter budget derivation is explicit (12,288 = 64 modules × 192 params/module).
- **Remaining weaknesses**:
  1. Right-side statistics are activation-proxy only (asymmetric bilateral story).
  2. Core empirical bet (JORA-base > LoRA-r1 at 2.3% of its parameter count) is unvalidated.
  3. Diag-only-selected may match JORA — rotation claim uncertain until experiments.
  4. Single dataset (Alpaca-cleaned) in main paper.
