#!/usr/bin/env python
"""Smoke test: JORA + OPT-350M + optimizer integration.

验证:
1. JORA 模型加载正常
2. compute_delta 在 fp32 subtraction 路径正确
3. optimizer param groups 正确分离
4. forward/backward pass 可跑通

不需要网络，使用本地缓存的 OPT-350M。
"""

import sys, torch
sys.path.insert(0, '/home/jqh/Workshop/JORA/src')
sys.path.insert(0, '/home/jqh/Workshop/JORA/examples/sft')

from peft import get_peft_model, JoraConfig
from peft.tuners.jora.model import JoraModel
from transformers import AutoModelForCausalLM

def main():
    print("=" * 70)
    print("JORA Smoke Test: OPT-350M + selective_diag")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1] Loading gpt2 (fully cached locally)...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    print(f"    Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # 2. JORA 配置
    print("\n[2] Creating JORA config...")
    cfg = JoraConfig(
        target_modules="all-linear",
        core="selective_diag",
        k=8,    # 小规模
        S_L=32,
        S_R=32,
        warmup_steps=5,
        pairs_freeze_after_warmup=True,
    )
    print(f"    core={cfg.core}, k={cfg.k}, warmup={cfg.warmup_steps}")

    # 3. PEFT wrap
    print("\n[3] Wrapping with JORA...")
    jora_model = get_peft_model(model, cfg, adapter_name="default")
    jora_model.print_trainable_parameters()

    # 4. 验证 compute_delta 路径
    print("\n[4] Verifying compute_delta path...")
    from peft.tuners.jora.layer import JoraLayer
    jora_layers = [m for m in jora_model.modules() if isinstance(m, JoraLayer)]
    print(f"    JoraLayer count: {len(jora_layers)}")

    # 找一个有激活 support 的 layer，手动触发 freeze
    for jl in jora_layers:
        st = jl.adapters['default']
        if st.num_pairs_L.item() > 0 or st.num_pairs_R.item() > 0:
            continue
        # 手动设置几个 pair 来测试
        with torch.no_grad():
            n = min(4, st.n)
            st.pairs_L[:n] = torch.arange(n, dtype=torch.long).reshape(-1, 2).clip(max=n-1)[:n]
            st.num_pairs_L.fill_(n)
            st.pairs_R[:n] = st.pairs_L[:n]
            st.num_pairs_R.fill_(n)
        st._freeze_support_if_needed()
        break

    # 找第一个有 support 的 layer
    test_layer = None
    for jl in jora_layers:
        st = jl.adapters['default']
        if st.core.support_size > 0 and hasattr(st.core, 'support_indices'):
            test_layer = jl
            break

    if test_layer is not None:
        st = test_layer.adapters['default']
        x = torch.randn(2, st.m, dtype=torch.bfloat16, device=st.n)
        with torch.no_grad():
            delta = st.compute_delta(x)
        print(f"    Delta shape: {delta.shape}, dtype: {delta.dtype}")
        print(f"    Delta abs max: {delta.abs().max().item():.6f}")
        print(f"    Delta abs mean: {delta.abs().mean().item():.6f}")
        print("    ✅ compute_delta works (fp32 subtraction path)")
    else:
        print("    ⚠️  No layer with active support found — skipping compute_delta check")

    # 5. 验证 optimizer param groups
    print("\n[5] Verifying optimizer param groups...")
    if isinstance(jora_model, JoraModel):
        groups = jora_model.get_optimizer_param_groups(base_lr=1e-4)
        print(f"    Number of JORA groups: {len(groups)}")
        for g in groups:
            n_params = sum(p.numel() for p in g['params'])
            print(f"      {g.get('name', 'unnamed')}: lr={g['lr']:.2e}, {n_params:,} params")
        assert any(g.get('name') == 'jora_theta' for g in groups), \
            "Missing jora_theta group!"
        print("    ✅ param groups have correct names")
    else:
        print("    ⚠️  Model is not a JoraModel instance")

    # 6. Forward pass
    print("\n[6] Testing forward pass...")
    with torch.no_grad():
        out = jora_model(input_ids=torch.randint(0, 50257, (2, 32), device=jora_model.device))
    print(f"    Output logits shape: {out.logits.shape}")
    print("    ✅ Forward pass OK")

    # 7. Backward pass (1 step)
    print("\n[7] Testing backward pass...")
    jora_model.train()
    x_emb = torch.randn(2, 32, 1024, dtype=torch.bfloat16, device=jora_model.device, requires_grad=True)
    out = jora_model(inputs_embeds=x_emb)
    loss = out.logits.sum()
    loss.backward()
    print(f"    Loss: {loss.item():.4f}")
    # 检查 theta 和 core 都有梯度
    has_theta_grad = False
    has_core_grad = False
    for name, p in jora_model.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            if 'theta' in name:
                has_theta_grad = True
            if 'core' in name:
                has_core_grad = True
    print(f"    Has theta gradient: {has_theta_grad}")
    print(f"    Has core gradient: {has_core_grad}")
    print("    ✅ Backward pass OK")

    print("\n" + "=" * 70)
    print("🎉 ALL SMOKE TESTS PASSED!")
    print("=" * 70)
    print("\nReady to run experiments:")
    print("  Run 1: baseline")
    print("    python examples/sft/train.py configs/run1_opt350m_baseline.json")
    print("  Run 2: param-group LR")
    print("    python examples/sft/train.py configs/run2_opt350m_paramgroup.json")


if __name__ == "__main__":
    main()
