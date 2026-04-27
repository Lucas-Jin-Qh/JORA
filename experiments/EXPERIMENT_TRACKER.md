# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Freeze method wording | JORA-Diag formula audit | internal | formula / merge / save-load pass-fail | MUST | DONE | Covered by `docs/FORMULA_AUDIT.md`; additive DiagCore and residualized Selective distinctions are now explicit |
| R002 | M0 | Freeze deployment wording | JORA-NoRot save/load + merge sanity | internal | pass-fail | MUST | TODO | Keep TODO until the relevant tests are explicitly rerun and recorded |
| R003 | M0 | Freeze selective appendix wording | JORA-Selective save/load + merge sanity | internal | pass-fail | MUST | TODO | Keep TODO until the relevant tests are explicitly rerun and recorded |
| R004 | M1 | Fast mechanism pass | JORA-Diag ON, seed 42, 1 epoch | train | train loss, token acc, runtime | MUST | DONE/PARTIAL | Existing evidence suggests ON ≈ OFF |
| R005 | M1 | Fast mechanism pass baseline | JORA-NoRot, seed 42, 1 epoch | train | train loss, token acc, runtime | MUST | DONE/PARTIAL | Existing evidence suggests ON ≈ OFF |
| R006 | M2 | Longer-horizon mechanism verdict | JORA-Diag ON, seed 42, 3 epochs | train | final loss, token acc, runtime, learning curve | MUST | PARTIAL/DONE | Existing run evidence should be audited and normalized |
| R007 | M2 | Longer-horizon mechanism verdict baseline | JORA-NoRot, seed 42, 3 epochs | train | final loss, token acc, runtime, learning curve | MUST | DONE | Completed. Final summary: train_loss=2.23737, mean_token_accuracy=0.51498, runtime=3314.94s, steps=9222. Matched against R006 for 3-epoch ON/OFF verdict. |
| R008 | M3 | Variance check if verdict is close | JORA-Diag ON, seed 7 | train | final metric, runtime | MUST-IF-NEEDED | TODO | Launch only if R006 vs R007 remains ambiguous |
| R009 | M3 | Variance check if verdict is close | JORA-Diag ON, seed 2023 | train | final metric, runtime | MUST-IF-NEEDED | TODO | Same gate as R008 |
| R010 | M3 | Variance check if verdict is close | JORA-NoRot, seed 7 | train | final metric, runtime | MUST-IF-NEEDED | TODO | Pair with R008 |
| R011 | M3 | Variance check if verdict is close | JORA-NoRot, seed 2023 | train | final metric, runtime | MUST-IF-NEEDED | TODO | Pair with R009 |
| R012 | M4 | Strong baseline check | Tuned LoRA comparable budget | train/eval | final metric, params, runtime | MUST | TODO | Run only after ON/OFF verdict is frozen |
| R013 | M4 | Strong baseline check | Tuned DoRA comparable budget | train/eval | final metric, params, runtime | MUST | TODO | Skip only if implementation is unstable/unavailable |
| R014 | M5 | Appendix Pareto point | JORA-Selective | train/eval | final metric vs params | NICE | TODO | Appendix-only |
| R015 | M5 | Appendix context anchor | Compact LoRA anchor for Pareto | train/eval | final metric vs params | NICE | TODO | Only if needed for the appendix plot |
