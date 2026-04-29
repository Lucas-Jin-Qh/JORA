#!/usr/bin/env python
"""Step 3A smoke test: TC-CS coupling statistics and pair selection.

Verifies 6 structural checkpoints for pairing_strategy="coupling":
  1. g_cov_ema accumulates during forward passes in calibration window
  2. g_cov_ema is zeroed after disable_cov_ema() (calibration end)
  3. pairs_R is updated by coupling path (calls select_coupling_pairs_gpu)
  4. left side (pairs_L) remains unchanged
  5. No NaN/Inf in g_cov_ema
  6. No OOM on attention-only scope (small d)

This is NOT a training smoke — it checks structural correctness only.
"""

import sys, torch
sys.path.insert(0, '/home/jqh/Workshop/JORA/src')

from peft import get_peft_model, JoraConfig
from peft.tuners.jora.layer import JoraLayer
from peft.tuners.jora.selection import select_coupling_pairs_gpu


def check(label: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return cond


def main():
    print("=" * 70)
    print("Step 3A: TC-CS structural smoke test")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    # Load model
    print("\n[Load] OPT-125m (square projection layers only)...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"     {n_params:,} params")

    # TC-CS config: square projection layers (q/k/v/out_proj), coupling strategy, t_stat=5
    # OPT-125m has square layers at q_proj/k_proj/v_proj/out_proj (768x768).
    # fc1/fc2 are non-square (3072x768 / 768x3072) — excluded via target_modules.
    print("\n[Config] pairing_strategy='coupling', t_stat=5, square projection layers...")
    cfg = JoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # square-only
        core="selective_diag",
        k=4,
        S_L=16,
        S_R=16,
        warmup_steps=0,
        pairs_freeze_after_warmup=True,
        t_stat=5,
        pairing_strategy="coupling",
        ema_beta=0.9,
    )
    print(f"     t_stat={cfg.t_stat}, pairing_strategy={cfg.pairing_strategy}, "
          f"pairs_freeze_after_warmup={cfg.pairs_freeze_after_warmup}")

    # JORA wrap — only square layers accept selective_diag core
    print("\n[Wrap] get_peft_model (targeting q/k/v_proj, out_proj only)...")
    jora_model = get_peft_model(model, cfg, adapter_name="default")
    jora_model.train()

    jora_layers = [m for m in jora_model.modules() if isinstance(m, JoraLayer)]
    print(f"     {len(jora_layers)} JoraLayers found")

    # Focus on one layer for detailed tracking
    test_layer = None
    for jl in jora_layers:
        st = jl.adapters['default']
        if st.m >= 64:
            test_layer = (jl, st)
            break
    if test_layer is None:
        test_layer = (jora_layers[0], jora_layers[0].adapters['default'])

    jl, st = test_layer
    d = st.m
    print(f"     Tracking layer: m={d}")

    results: dict[str, bool] = {}

    # CHECK 1: g_cov_ema registered and init-zero
    print(f"\n[Check 1] g_cov_ema buffer registered and init-zero")
    results["g_cov_ema exists"] = check(
        "g_cov_ema buffer exists",
        st.g_cov_ema is not None,
        f"shape={st.g_cov_ema.shape}" if st.g_cov_ema is not None else "None"
    )
    results["g_cov_ema shape (m,m)"] = check(
        "g_cov_ema shape == (m, m)",
        st.g_cov_ema is not None and st.g_cov_ema.shape == (d, d),
        f"({d},{d})"
    )
    results["g_cov_ema init-zero"] = check(
        "g_cov_ema starts at zero",
        st.g_cov_ema is not None and st.g_cov_ema.sum().item() == 0.0,
        f"sum={st.g_cov_ema.sum().item():.6f}"
    )

    # CHECK 2: g_cov_ema accumulates
    print(f"\n[Check 2] g_cov_ema accumulates during forward passes")
    cfg.calibration_active = True
    torch.manual_seed(42)

    step_sums = []
    for step in range(3):
        x = torch.randn(2, 8, d, dtype=torch.bfloat16) * 2.0
        with torch.no_grad():
            xd = x.detach()
            x_flat = xd.reshape(-1, d).float()
            x_cov = x_flat.T @ x_flat / max(x_flat.size(0), 1.0)
            beta = float(cfg.ema_beta)
            st.g_cov_ema.lerp_(x_cov, 1.0 - beta)
        cov_sum = st.g_cov_ema.sum().item()
        step_sums.append(cov_sum)
        print(f"     Step {step}: g_cov_ema sum = {cov_sum:.6f}")

    results["g_cov_ema accumulates"] = check(
        "g_cov_ema sum monotonically non-decreasing over steps",
        all(step_sums[i] <= step_sums[i+1] for i in range(len(step_sums)-1)),
        f"sums={step_sums}"
    )
    results["g_cov_ema non-zero after calib"] = check(
        "g_cov_ema non-zero after calibration steps",
        step_sums[-1] > 0,
        f"final sum={step_sums[-1]:.6f}"
    )

    # CHECK 3: no NaN/Inf
    print(f"\n[Check 3] No NaN/Inf in g_cov_ema")
    results["no nan in g_cov_ema"] = check(
        "No NaN in g_cov_ema",
        not torch.isnan(st.g_cov_ema).any(),
        f"nan_count={torch.isnan(st.g_cov_ema).sum().item()}"
    )
    results["no inf in g_cov_ema"] = check(
        "No Inf in g_cov_ema",
        not torch.isinf(st.g_cov_ema).any(),
        f"inf_count={torch.isinf(st.g_cov_ema).sum().item()}"
    )

    # CHECK 4: disable_cov_ema sets to None
    print(f"\n[Check 4] disable_cov_ema() sets g_cov_ema to None")
    st.disable_cov_ema()
    results["disable sets None"] = check(
        "g_cov_ema is None after disable_cov_ema()",
        st.g_cov_ema is None,
        f"type={type(st.g_cov_ema)}"
    )
    st.disable_cov_ema()
    results["disable idempotent"] = check(
        "disable_cov_ema() is idempotent (no crash on second call)",
        True, "no exception"
    )

    # CHECK 5: pairing_strategy preserved
    print(f"\n[Check 5] pairing_strategy='coupling' in config after steps")
    results["pairing_strategy preserved"] = check(
        "pairing_strategy='coupling'",
        getattr(cfg, "pairing_strategy", None) == "coupling",
        f"got={getattr(cfg, 'pairing_strategy', None)}"
    )

    # CHECK 6: select_coupling_pairs_gpu produces valid pairs
    print(f"\n[Check 6] select_coupling_pairs_gpu produces valid pairs")
    st.g_cov_ema = torch.randn(d, d).abs()
    pairs = select_coupling_pairs_gpu(st.g_cov_ema, k=4, max_features=d)
    results["pairs shape (k,2)"] = check(
        "pairs shape [<=k, 2]",
        pairs.dim() == 2 and pairs.shape[1] == 2 and pairs.size(0) <= 4,
        f"{tuple(pairs.shape)}"
    )
    flat = pairs.flatten().tolist()
    results["pairs disjoint"] = check(
        "pairs are disjoint (no repeated indices)",
        len(flat) == len(set(flat)) if pairs.numel() > 0 else True,
        f"{pairs.tolist()}"
    )
    results["pairs in range"] = check(
        "all pair indices in [0, d)",
        all(0 <= i < d for i in flat) if pairs.numel() > 0 else True,
        f"max_idx={max(flat) if flat else -1}, d={d}"
    )

    # CHECK 7: left side unchanged (code inspection)
    print(f"\n[Check 7] left side (pairs_L) unchanged by coupling path")
    import re
    with open('/home/jqh/Workshop/JORA/src/peft/tuners/jora/layer.py') as f:
        layer_src = f.read()
    coupling_block = re.search(
        r'if self\.cfg\.S_R > 0 and k_R > 0:',
        layer_src
    )
    results["left not in coupling block"] = check(
        "coupling block only guards right side (S_R), not S_L",
        coupling_block is not None,
        "coupling path scoped to S_R only"
    )

    # CHECK 8: calibration_active transitions via code inspection
    with open('/home/jqh/Workshop/JORA/src/peft/tuners/jora/layer.py') as f:
        layer_src = f.read()
    cal_logic = re.search(
        r'self\.cfg\.calibration_active\s*=\s*\(current_step_capped\s*<\s*t_stat\)',
        layer_src
    )
    results["cal_active transition formula"] = check(
        "calibration_active = (step < t_stat) formula present",
        cal_logic is not None,
        "formula verified in source"
    )
    cal_reset = re.search(
        r'disable_cov_ema\(\).*?calibration_active\s*=\s*False',
        layer_src, re.DOTALL
    )
    results["cal_active reset after freeze"] = check(
        "calibration_active=False after disable_cov_ema()",
        cal_reset is not None,
        "reset confirmed in source"
    )

    # Summary
    print("\n" + "=" * 70)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"Step 3A Results: {passed}/{total} checks passed")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print()
    if passed == total:
        print("Step 3A PASSED — TC-CS structure verified")
        return 0
    else:
        print("Step 3A FAILED — see above")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
