# JORA Agent Contract

## Project
- Project: JORA, PEFT method for LLM fine-tuning.
- Current main method: JORA-Diag.
- Current risk: rotation contribution is unproven; JORA-NoRot is a first-class baseline.

## Hard Rules
- Do not claim "rotation drives the gain" unless JORA-Diag ON beats JORA-NoRot on matched eval.
- Do not write final Method formula until FORMULA_AUDIT.md is complete.
- Do not run full expensive sweeps before sanity and save/load/merge checks pass.
- Do not modify unrelated PEFT methods unless required for baseline compatibility.

## Environment
- OS: Linux (6.8.0-40-generic)
- Python: /home/jqh/miniconda3/envs/peft-jora/bin/python
- CUDA: available
- PyTorch: 2.8.0+cu128
- Conda activate: conda activate peft-jora
- Main repo path: /home/jqh/Workshop/JORA
- HF mirror: HF_ENDPOINT=https://hf-mirror.com
- GPU inventory: RTX 5090 (3 cards, all free), HF_ENDPOINT+TRANSFORMERS_OFFLINE=1
- code_sync: git
- wandb: offline mode, project=jora

## Canonical Commands
- train JORA-Diag: `python scripts/run_jora_exp.py configs/run_diag_main_s42_3ep.json --gpu 0`
- train JORA-NoRot: `python scripts/run_jora_exp.py configs/run_diag_no_rotation_s42_3ep.json --gpu 0`
- eval MMLU: (not yet in canonical form)
- eval ARC-C: (not yet in canonical form)
- eval GSM8K: (not yet in canonical form)
- parameter count script: (see HANDOFF.md)
- save/load roundtrip test: (see HANDOFF.md)
- merge equivalence test: (see HANDOFF.md)
- smoke test: `/home/jqh/miniconda3/envs/peft-jora/bin/python -c "..."` (see HANDOFF.md)

## Current Evidence
- 1ep JORA-Diag ON train_loss = 2.23870, runtime = 52 min.
- 1ep JORA-NoRot train_loss = 2.23866, runtime = 17 min.
- 3ep JORA-Diag ON train_loss = 2.2374, token_acc = 0.5147, runtime = ~158 min.
- 3ep JORA-NoRot train_loss = 2.2368, token_acc = 0.5150, runtime = ~55 min.
- **TC-CS normalized correlation (Step 4.8): pair overlap = 100% vs consecutive, train_loss delta = 2.7e-6 (noise level). Gate FAIL — TC-CS degenerates to consecutive pairing in real training.**
- **M0 correctness gate (save/load + merge): PASS.** Exact merge via basis-probing for both SelectiveDiagCore and DiagCore (C1.6 fix preserved after Option C revert). DiagCore additive formula: Δ(x) = R_L^T Diag(d) R_R x.
- **Option C (residualized DiagCore): CATASTROPHIC FAIL.** 1ep eval loss = 15.7 (ON) and 19.2 (NoRot) vs base 5.5 and additive JORA-Diag 2.24. Root cause: R_L^T @ R_R != I when left/right pairs are independent. DiagCore reverted to additive default (2026-04-28). See `docs/JORA_OPTION_C_POSTMORTEM.md`.
- Interpretation: no discernible train-loss gain from rotation OR coupling strategy; runtime cost high. M0 correctness closed. Next: M1 LoRA/DoRA baselines.