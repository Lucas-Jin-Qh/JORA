# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | sanity | TC-CS-1S config parse + single_sided="right" wiring | internal | pass/fail | MUST | **GATE FAILED** | **Mechanism gate FAIL (Step 4.8): pair overlap = 100% (384/384) on real OPT-350m activations; 100-step loss delta = 2.7e-6 (noise); requires candidate-pool redesign before further training experiments** |
| R002 | M0 | sanity | forward-pass covariance EMA accumulation | internal | pass/fail, buffer shape, memory | MUST | **PAUSED** | Paused — TC-CS mechanism gate failed; covariance EMA infrastructure is valid but candidate pool needs redesign |
| R003 | M0 | sanity | subspace freeze + pair restriction after calibration | internal | pass/fail | MUST | **PAUSED** | Paused — TC-CS mechanism gate failed; re-enable after candidate-pool redesign |
| R004 | M1 | baseline anchor | JORA-NoRot matched reference | train | final quality, loss, token acc, runtime | MUST | TODO/REUSE | Reuse only if reporting protocol matches the new plan |
| R005 | M1 | baseline anchor | current additive JORA-Diag matched reference | train | final quality, loss, token acc, runtime | MUST | TODO/REUSE | Reuse only if reporting protocol matches the new plan |
| R006 | M2 | main mechanism test | TC-CS-1S attention-only | train/eval | final quality, loss, token acc, runtime, step time | MUST | **BLOCKED** | BLOCKED — mechanism gate failed (R001 FAIL); candidate-pool redesign required before R006 can proceed |
| R007 | M2 | main mechanism comparison | JORA-NoRot | train/eval | final quality, loss, token acc, runtime, step time | MUST | TODO/REUSE | Matched comparator for R006 |
| R008 | M2 | main mechanism comparison | current additive JORA-Diag | train/eval | final quality, loss, token acc, runtime, step time | MUST | TODO/REUSE | Status-quo rotation comparator for R006 |
| R009 | M3 | coupling regime check | TC-CS-1S FFN-only | train/eval | final quality, runtime | MUST | **BLOCKED** | BLOCKED — depends on R006 passing |
| R010 | M3 | coupling regime check | TC-CS-1S all-linear | train/eval | final quality, runtime | MUST | **BLOCKED** | BLOCKED — depends on R006 passing |
| R011 | M4 | simplicity check | compare one-sided TC-CS-1S against current broader global rotation | analysis | relative quality + runtime | MUST | **BLOCKED** | BLOCKED — depends on R006 passing |
| R012 | M5 | diagnostic | theta norm / grad diagnostics for TC-CS-1S | train | diagnostic summaries | NICE | **CANCELLED** | Cancelled — TC-CS mechanism gate failed; no rotation diagnostic needed |
| R013 | M5 | diagnostic | subspace overlap across seeds | analysis | overlap / stability | NICE | **CANCELLED** | Cancelled — TC-CS mechanism gate failed; subspace does not differentiate on real activations |
