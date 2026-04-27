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
- OS:
- Python:
- CUDA:
- PyTorch:
- Conda activate:
- Main repo path:
- Remote server SSH:
- GPU inventory:
- code_sync: git or rsync
- wandb: true/false
- wandb_project: jora

## Canonical Commands
- train JORA-Diag:
- train JORA-NoRot:
- eval MMLU:
- eval ARC-C:
- eval GSM8K:
- parameter count script:
- save/load roundtrip test:
- merge equivalence test:

## Current Evidence
- 1ep JORA-Diag ON train_loss = 2.23870, runtime = 52 min.
- 1ep JORA-NoRot train_loss = 2.23866, runtime = 17 min.
- Interpretation: no discernible 1ep train-loss gain from rotation; runtime cost high.