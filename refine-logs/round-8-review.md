# Round 8 Review — GPT-5.4 xhigh

**ThreadId**: `019cfef1-9bb6-7570-ba01-7f2c18b7eb58`
**Date**: 2026-03-18
**Verdict**: REVISE
**Overall Score**: 8/10

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8 |
| Method Specificity | 8 |
| Contribution Quality | 8 |
| Frontier Leverage | 8 |
| Feasibility | 8 |
| Validation Focus | 9 |
| Venue Readiness | 7 |
| **Overall** | **8** |

## Anchor Check
**Preserved.** The proposal still attacks the original anchored problem: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The scope is narrower and cleaner, not drifted.

## Assessment
- **Dominant contribution**: sharp. Clearly one mechanism: one-shot adaptive allocation of sparse bilateral rotation support inside rotation-based PEFT.
- **Method size**: simple and no longer overbuilt.
- **Frontier leverage**: appropriate. Nothing feels old-school or trend-forced.
- **Concrete enough from the algorithm box alone**: almost, but not literally yet. The post-allocation adapter is implementable. The remaining gap is the calibration block, which mixes `torch.no_grad()` language with gradient extraction and should be rewritten as a full-model forward/backward calibration pass with layer hooks.

## Remaining Action Items

### IMPORTANT
1. **Algorithm 1 calibration pseudocode is internally inconsistent**: Cannot collect `g = autograd(loss, y_base)` inside `torch.no_grad()`, and `task_loss(y_base)` reads like a local-layer loss rather than a full-model loss. Fix: rewrite calibration as a separate pre-training calibration pass with `no optimizer step`, full-model loss, and output hooks — no `torch.no_grad()` wrapper around the backward.

2. **δ implementation form underspecified**: Specify explicitly whether δ is stored as a packed support vector plus scatter, or as a full-length masked tensor. Both work; the choice affects memory and backward implementation.

3. **Claim 2 conditionality**: State explicitly that claim 2 (JORA beats qGOFT) is made only if faithful qGOFT reproduction succeeds; otherwise it becomes an internal static-baseline comparison only.

### MINOR
4. **Right-side bilateral claim**: Keep the bilateral claim narrow and factual: output side is gradient-guided, input side is activation-proxy-guided. Do not oversell.

5. **Diag-only-selected is claim-determining**: Treat it as claim-determining, not just diagnostic. If within the precommitted margin, narrow the paper accordingly.

## Simplification Opportunities
- Rewrite calibration as a separate pre-training calibration pass with `no optimizer step`, full-model loss, and output hooks.
- If ablations show little value from right-side adaptive allocation, simplify to left-side adaptive + fixed right-side support.
- Keep JORA-large, non-residualized init, and breadth baselines appendix-only.

## Modernization Opportunities
NONE

## Drift Warning
NONE

<details>
<summary>Full Raw Review</summary>

**Problem Anchor**

`Preserved.` The proposal still attacks the original anchored problem: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The scope is narrower and cleaner, not drifted.

**Scores**

1. `Problem Fidelity`: **8/10**
2. `Method Specificity`: **8/10**
3. `Contribution Quality`: **8/10**
4. `Frontier Leverage`: **8/10**
5. `Feasibility`: **8/10**
6. `Validation Focus`: **9/10**
7. `Venue Readiness`: **7/10**

**OVERALL SCORE**: **8/10**

**Assessment**

`Dominant contribution:` sharp. The paper is now clearly about one mechanism: one-shot adaptive allocation of sparse bilateral rotation support inside rotation-based PEFT.

`Method size:` simple and no longer overbuilt.

`Frontier leverage:` appropriate. This is modern enough already; nothing feels old-school or trend-forced.

`Concrete enough from the algorithm box alone:` almost, but not literally yet. The post-allocation adapter is implementable from the box. The remaining gap is the calibration block, which still mixes `torch.no_grad()` language with gradient extraction and should be rewritten as a full-model forward/backward calibration pass with layer hooks.

**Main Remaining Critiques**

- `Missing mechanism / integration point:` Algorithm 1 calibration is internally inconsistent. You cannot collect `g = autograd(loss, y_base)` inside `torch.no_grad()`, and `task_loss(y_base)` reads like a local-layer loss rather than a full-model loss. This is the main remaining specificity gap.
- `Weak training signal:` the right side is still only an activation proxy. That is acceptable, but it means the bilateral story remains asymmetric. If `fixed-slot JORA` and a left-only-adaptive variant are close, the right-side allocation is not carrying much real signal.
- `Pseudo-novelty risk:` the paper is strong only if `Diag-only-selected` is materially worse than JORA. Your precommitted interpretation rule is good; the paper should actually obey it.
- `Prior-art dependence:` claim 2 is only as strong as the faithful `qGOFT` reproduction. If that baseline is not clean, the published-baseline claim should be dropped rather than replaced rhetorically by `fixed-slot JORA`.

**Verdict**

`REVISE`

**Drift Warning**

`NONE`

**Simplification Opportunities**

- Rewrite calibration as a separate pre-training calibration pass with `no optimizer step`, full-model loss, and output hooks. That removes the last integration ambiguity.
- If ablations show little value from right-side adaptive allocation, simplify to left-side adaptive allocation plus fixed right-side support.
- Keep `JORA-large`, non-residualized init, and breadth baselines appendix-only.

**Modernization Opportunities**

`NONE`

**Remaining Action Items**

1. Fix Algorithm 1 calibration pseudocode so it is literally implementable: no `torch.no_grad()` around gradient collection, and define hook-based `ema_out` accumulation from a full-model backward pass.
2. Specify the implementation form of `δ` on `U` unambiguously: packed support vector plus scatter, or full-length masked tensor.
3. State explicitly that claim 2 is only made if faithful `qGOFT` reproduction succeeds; otherwise it becomes an internal static-baseline comparison only.
4. Keep the bilateral claim narrow and factual: output side is gradient-guided, input side is activation-proxy-guided.
5. Treat `Diag-only-selected` as claim-determining, not just diagnostic. If it is within the precommitted margin, narrow the paper accordingly.

</details>
