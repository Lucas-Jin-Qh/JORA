# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | sanity — pipeline gate | JORA-base (K=32), OPT-350M, 500 steps, seed=42 | alpaca-cleaned train; MMLU test | loss ↓, param_count=192×N_modules, MMLU runs | MUST | TODO | Verify loss decreasing; check exact param count |
| R002 | M0 | sanity — LoRA baseline | LoRA-r1, OPT-350M, 500 steps, seed=42 | alpaca-cleaned train; MMLU test | loss ↓, MMLU accuracy | MUST | TODO | Cross-check MMLU scorer correctness |
| R003 | M1 | LR sweep — JORA theta | JORA-base, OPT-350M, lr_theta ∈ {1e-3,5e-3,1e-2,5e-2} × lr_core=1e-3, 500 steps, seed=42 | alpaca-cleaned train; MMLU 200-sample | MMLU acc, final loss | MUST | TODO | 4-run sub-grid; identify best lr_theta |
| R004 | M1 | LR sweep — JORA core | JORA-base, OPT-350M, best lr_theta × lr_core ∈ {5e-4,1e-3,5e-3}, 500 steps, seed=42 | alpaca-cleaned train; MMLU 200-sample | MMLU acc, final loss | MUST | TODO | 3-run sub-grid; finalize (lr_theta, lr_core) |
| R005 | M2 | 1-seed screening — JORA-base | JORA-base (K=32), Mistral-7B-v0.1, 2000 steps, seed=42 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy, param count | MUST | TODO | Uses best LR from R003-R004 |
| R006 | M2 | 1-seed screening — LoRA-r1 | LoRA-r1 (r=1, ~524K params), Mistral-7B-v0.1, 2000 steps, seed=42 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | Gate: JORA-base > LoRA-r1 directionally |
| R007 | M2 | 1-seed screening — LoRA-r2 | LoRA-r2 (r=2, ~1M params), Mistral-7B-v0.1, 2000 steps, seed=42 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | Primary comparison; check 2pp gap |
| R008 | M2 | 1-seed screening — fixed-slot (B3) | Fixed-slot JORA (selection=random, K=32), Mistral-7B-v0.1, 2000 steps, seed=42 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | Adaptive vs static directional signal |
| R009 | M2 | 1-seed screening — diag-only (B4) | Diag-only-selected (same U from R005, θ=0 frozen, only δ trained), Mistral-7B-v0.1, 2000 steps, seed=42 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | Rotation claim: JORA-base vs diag-only |
| R010 | M3 | multi-seed — JORA-base seed 1 | JORA-base (K=32), Mistral-7B-v0.1, 2000–5000 steps, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy ± std | MUST | TODO | Run after M2 gate passes |
| R011 | M3 | multi-seed — JORA-base seed 2 | JORA-base (K=32), Mistral-7B-v0.1, same steps, seed=2 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R012 | M3 | multi-seed — JORA-base seed 3 | JORA-base (K=32), Mistral-7B-v0.1, same steps, seed=3 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R013 | M3 | multi-seed — LoRA-r2 seed 1 | LoRA-r2 (r=2), Mistral-7B-v0.1, same steps, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy ± std | MUST | TODO | |
| R014 | M3 | multi-seed — LoRA-r2 seed 2 | LoRA-r2 (r=2), Mistral-7B-v0.1, same steps, seed=2 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R015 | M3 | multi-seed — LoRA-r2 seed 3 | LoRA-r2 (r=2), Mistral-7B-v0.1, same steps, seed=3 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R016 | M3 | multi-seed — LoRA-r1 seed 1 | LoRA-r1 (r=1), Mistral-7B-v0.1, same steps, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | 2 seeds sufficient for LoRA-r1 |
| R017 | M3 | multi-seed — LoRA-r1 seed 2 | LoRA-r1 (r=1), Mistral-7B-v0.1, same steps, seed=2 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R018 | M3 | multi-seed — JORA-small seed 1 | JORA-small (K=8, ~3K params), Mistral-7B-v0.1, same steps, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | 2 seeds for Pareto anchor |
| R019 | M3 | multi-seed — JORA-small seed 2 | JORA-small (K=8, ~3K params), Mistral-7B-v0.1, same steps, seed=2 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R020 | M3 | multi-seed — LoRA-r4 seed 1 | LoRA-r4 (r=4, ~2M params), Mistral-7B-v0.1, same steps, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | Upper anchor; 2 seeds |
| R021 | M3 | multi-seed — LoRA-r4 seed 2 | LoRA-r4 (r=4, ~2M params), Mistral-7B-v0.1, same steps, seed=2 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R022 | M3 | multi-seed — fixed-slot seed 1 (B3) | Fixed-slot JORA (selection=random, K=32), Mistral-7B-v0.1, same steps, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R023 | M3 | multi-seed — fixed-slot seed 2 (B3) | Fixed-slot JORA (selection=random, K=32), Mistral-7B-v0.1, same steps, seed=2 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R024 | M3 | multi-seed — fixed-slot seed 3 (B3) | Fixed-slot JORA (selection=random, K=32), Mistral-7B-v0.1, same steps, seed=3 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R025 | M3 | multi-seed — diag-only seed 1 (B4) | Diag-only-selected, Mistral-7B-v0.1, same steps, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | Reuse U slots from corresponding JORA-base run |
| R026 | M3 | multi-seed — diag-only seed 2 (B4) | Diag-only-selected, Mistral-7B-v0.1, same steps, seed=2 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | MUST | TODO | |
| R027 | M4 | diagnostics — Jaccard curves (B5) | JORA-base logs from R010–R012; compute Jaccard(top-2K indices, t, t-1) | calibration window (first T_stat steps) | Jaccard per layer, averaged | MUST | TODO | Post-hoc from B1 logs; log ema_in snapshots every 20 steps |
| R028 | M4 | diagnostics — qGOFT QA (B3 condition) | qGOFT reimplementation in PEFT, ~12K params, Mistral-7B | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | loss convergence ≤110% LoRA-r1; param count within 5% of 12K | MUST | TODO | QA pass/fail decision before claiming C2 vs qGOFT |
| R029 | M5 | appendix — T_stat sensitivity (B6) | JORA-base T_stat ∈ {50,100,200,400}, Mistral-7B, seed=1 | alpaca-cleaned train; MMLU/ARC-C/GSM8K test | avg accuracy | NICE | TODO | 4 runs; plateau check |
| R030 | M5 | appendix — non-residualized init (B7) | JORA-base residualized vs non-residualized (D_sel=I_U), Mistral-7B, seed=1, 500 steps | alpaca-cleaned train; MMLU test | loss at steps 1–10 post-allocation | NICE | TODO | Loss spike > 0.1 nats expected in non-residualized |

## Status Key

- **TODO** — not started
- **RUNNING** — currently executing
- **DONE** — finished, results available
- **FAILED** — run failed or quarantined; needs investigation
- **SKIP** — decided to skip (reason in Notes)

## Quarantine Policy

Any run with final MMLU accuracy below the pretrained baseline (OPT-350M: ~0.26; Mistral-7B: ~0.60) is flagged FAILED and must not be included in paper tables without a documented explanation.

## Results Summary (fill as runs complete)

| Run ID | MMLU | ARC-C | GSM8K | Avg | Notes |
|--------|------|-------|-------|-----|-------|
| R001 | — | — | — | — | |
| R002 | — | — | — | — | |
| R005 | — | — | — | — | |
| R006 | — | — | — | — | |
| R007 | — | — | — | — | |
