"""Tests for the paper-exact JORA path (SelectiveDiagCore, calibration, merge).

Run with: pytest tests/test_jora_paper_path.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from peft.tuners.jora.core import SelectiveDiagCore, build_core
from peft.tuners.jora.config import JoraConfig


# ------------------------------------------------------------------ helpers --

def _make_cfg(**kwargs) -> JoraConfig:
    defaults = dict(
        target_modules=["q_proj"],
        core="selective_diag",
        magnitude="none",
        zero_init_core=True,
        pairs_freeze_after_warmup=True,
        theta_init_std=0.0,
        k=8,
        S_L=16,
        S_R=16,
        rotation_impl="torch",  # avoid Triton in unit tests
    )
    defaults.update(kwargs)
    return JoraConfig(**defaults)


# ------------------------------------------- SelectiveDiagCore unit tests ---

class TestSelectiveDiagCore:
    def test_param_count(self):
        """SelectiveDiagCore(k=8) stores exactly 2k = 16 scalar params."""
        core = SelectiveDiagCore(support_size=16, device="cpu", dtype=torch.float32)
        assert core.delta.numel() == 16
        total_trainable = sum(p.numel() for p in core.parameters())
        assert total_trainable == 16

    def test_zero_init(self):
        """delta is initialised to zero."""
        core = SelectiveDiagCore(support_size=16, device="cpu", dtype=torch.float32)
        assert core.delta.abs().max().item() == 0.0

    def test_apply_to_vector_zero_delta(self):
        """When delta=0, apply_to_vector projects x onto U (not identity on full space)."""
        core = SelectiveDiagCore(support_size=4, device="cpu", dtype=torch.float32)
        indices = torch.tensor([0, 2, 4, 6])
        core.set_support(indices)
        x = torch.ones(8)
        y = core.apply_to_vector(x)
        # y[U] = x[U] * (1 + 0) = 1,  y[~U] = 0
        assert y[0].item() == pytest.approx(1.0)
        assert y[2].item() == pytest.approx(1.0)
        assert y[4].item() == pytest.approx(1.0)
        assert y[6].item() == pytest.approx(1.0)
        assert y[1].item() == pytest.approx(0.0)
        assert y[3].item() == pytest.approx(0.0)

    def test_apply_to_vector_nonzero_delta(self):
        """apply_to_vector correctly scales by (1+delta) on U."""
        core = SelectiveDiagCore(support_size=2, device="cpu", dtype=torch.float32)
        core.set_support(torch.tensor([1, 3]))
        with torch.no_grad():
            core.delta.fill_(0.5)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = core.apply_to_vector(x)
        assert y[0].item() == pytest.approx(0.0)
        assert y[1].item() == pytest.approx(2.0 * 1.5)
        assert y[2].item() == pytest.approx(0.0)
        assert y[3].item() == pytest.approx(4.0 * 1.5)

    def test_apply_to_vector_preserves_bf16_dtype(self):
        """SelectiveDiagCore must not promote bf16 activations to float during indexed writes."""
        core = SelectiveDiagCore(support_size=2, device="cpu", dtype=torch.bfloat16)
        core.set_support(torch.tensor([0, 2]))
        with torch.no_grad():
            core.delta.fill_(0.5)
        x = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.bfloat16)
        y = core.apply_to_vector(x)

        assert y.dtype == torch.bfloat16
        expected = torch.tensor([3.0, 0.0, 6.0, 0.0], dtype=torch.bfloat16)
        torch.testing.assert_close(y, expected, rtol=0.0, atol=0.0)

    def test_project_support(self):
        """project_support keeps only U-indexed entries."""
        core = SelectiveDiagCore(support_size=3, device="cpu", dtype=torch.float32)
        core.set_support(torch.tensor([0, 2, 4]))
        x = torch.arange(6, dtype=torch.float32)
        px = core.project_support(x)
        assert px[0].item() == pytest.approx(0.0)
        assert px[2].item() == pytest.approx(2.0)
        assert px[4].item() == pytest.approx(4.0)
        assert px[1].item() == pytest.approx(0.0)
        assert px[3].item() == pytest.approx(0.0)
        assert px[5].item() == pytest.approx(0.0)

    def test_set_support_size_check(self):
        """set_support raises if >support_size indices passed."""
        core = SelectiveDiagCore(support_size=4, device="cpu", dtype=torch.float32)
        with pytest.raises(AssertionError):
            core.set_support(torch.tensor([0, 1, 2, 3, 4]))  # 5 > 4

    def test_set_support_partial_ok(self):
        """set_support accepts fewer than support_size indices (no aliasing)."""
        core = SelectiveDiagCore(support_size=8, device="cpu", dtype=torch.float32)
        idxs = torch.tensor([0, 2, 4])  # only 3 out of 8
        core.set_support(idxs)  # must not raise
        assert core._active_support_size_py == 3
        # Tail delta entries must be zeroed to avoid aliasing
        assert core.delta[3:].abs().max().item() == 0.0

    def test_num_params_property(self):
        core = SelectiveDiagCore(support_size=10, device="cpu", dtype=torch.float32)
        assert core.num_params == 10


# ------------------------------------------- build_core factory tests -------

class TestBuildCore:
    def test_build_selective_diag(self):
        cfg = _make_cfg(k=8)
        core = build_core("selective_diag", n=64, m=64, device="cpu", dtype=torch.float32, cfg=cfg)
        assert isinstance(core, SelectiveDiagCore)
        assert core.support_size == 16  # 2k = 16

    def test_build_selective_diag_param_count(self):
        cfg = _make_cfg(k=4)
        core = build_core("selective_diag", n=32, m=32, device="cpu", dtype=torch.float32, cfg=cfg)
        assert core.delta.numel() == 8  # 2 * 4


# ------------------------------------------- JoraConfig factory tests -------

class TestJoraConfigPaperPath:
    def test_paper_path_factory_defaults(self):
        cfg = JoraConfig.paper_path(target_modules=["q_proj"])
        assert cfg.core == "selective_diag"
        assert cfg.magnitude == "none"
        assert cfg.zero_init_core is True
        assert cfg.pairs_freeze_after_warmup is True
        assert cfg.theta_init_std == 0.0

    def test_paper_path_factory_override(self):
        cfg = JoraConfig.paper_path(target_modules=["q_proj"], k=16, magnitude="ecd_tanh")
        assert cfg.k == 16
        assert cfg.magnitude == "ecd_tanh"  # overridden
        assert cfg.core == "selective_diag"  # still default

# ------------------------------------------- Zero-function-change at init ---

class TestZeroFunctionChange:
    """At initialisation, the JORA adapter should not change the output."""

    def _build_layer(self, in_f: int = 32, out_f: int = 32, k: int = 4):
        from peft.tuners.jora.layer import JoraLayer

        base = nn.Linear(in_f, out_f, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=k,
            S_L=min(k * 2, out_f),
            S_R=min(k * 2, in_f),
            rotation_impl="torch",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        return layer, base, layer.adapters["default"]

    def test_zero_change_before_set_support(self):
        """Before set_support is called, adapter output equals base output."""
        layer, base, _ = self._build_layer(in_f=32, out_f=32)
        x = torch.randn(4, 32)
        with torch.no_grad():
            out_layer = layer(x)
            out_base = base(x)
        assert torch.allclose(out_layer, out_base, atol=1e-5), (
            f"Max diff: {(out_layer - out_base).abs().max().item()}"
        )


# ------------------------------------------- Frozen support after freeze ----

class TestFrozenSupport:
    def test_constructor_smoke(self):
        """JoraLayer constructor must succeed without AttributeError."""
        from peft.tuners.jora.layer import JoraLayer

        base = nn.Linear(16, 16, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=4,
            S_L=8,
            S_R=8,
            rotation_impl="torch",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]
        # Both Python flag and buffer must be initialized
        assert hasattr(adapter, "_pairs_frozen"), "_pairs_frozen not initialized"
        assert hasattr(adapter, "pairs_frozen_flag"), "pairs_frozen_flag buffer missing"
        assert adapter._pairs_frozen is False
        assert not bool(adapter.pairs_frozen_flag.item())

    def test_freeze_sets_frozen_flag(self):
        from peft.tuners.jora.layer import JoraLayer

        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=4,
            S_L=8,
            S_R=8,
            selection="topk_ema",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]

        assert not getattr(adapter, "_pairs_frozen", False)
        assert not bool(adapter.pairs_frozen_flag.item()), "pairs_frozen_flag should start False"
        adapter._freeze_support_if_needed()
        assert adapter._pairs_frozen is True
        assert bool(adapter.pairs_frozen_flag.item()), "pairs_frozen_flag should be True after freeze"

    def test_freeze_idempotent(self):
        """Calling _freeze_support_if_needed() twice should not raise."""
        from peft.tuners.jora.layer import JoraLayer

        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=4,
            S_L=8,
            S_R=8,
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]
        adapter._freeze_support_if_needed()
        adapter._freeze_support_if_needed()
        assert adapter._pairs_frozen is True

    def test_update_step_does_not_mutate_after_freeze(self):
        """After freeze, update_step() must not mutate pairs_L/pairs_R."""
        from peft.tuners.jora.layer import JoraLayer

        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=4,
            S_L=8,
            S_R=8,
            selection="topk_ema",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]

        adapter._freeze_support_if_needed()
        assert adapter._pairs_frozen is True

        pairs_L_before = adapter.pairs_L.clone()
        pairs_R_before = adapter.pairs_R.clone()

        # Simulate more steps — should be no-ops
        adapter.update_step(50, 100)
        adapter.update_step(100, 100)

        assert torch.equal(pairs_L_before, adapter.pairs_L), "pairs_L mutated after freeze"
        assert torch.equal(pairs_R_before, adapter.pairs_R), "pairs_R mutated after freeze"


# ------------------------------------------- Merge equivalence --------------

class TestMergeEquivalence:
    """Merged weight should apply the same linear map as the adapter forward pass.

    Note: selective_diag (paper path) requires square layers (in_features == out_features).
    Rectangular layers are intentionally skipped in JoraModel._create_and_replace().
    """

    @pytest.mark.parametrize("d", [16, 32])
    def test_merge_equals_forward(self, d: int):
        """Merge is exact for zero theta (square layers only)."""
        from peft.tuners.jora.layer import JoraLayer

        torch.manual_seed(42)
        base = nn.Linear(d, d, bias=False)
        k = 4
        sl = k * 2
        sr = k * 2
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=k,
            S_L=sl,
            S_R=sr,
            selection="none",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]

        with torch.no_grad():
            idxs = torch.arange(k * 2, dtype=torch.long)
            adapter.core.set_support(idxs)
            adapter.core.delta.uniform_(-0.3, 0.3)

        x = torch.randn(8, d)

        with torch.no_grad():
            out_adapter = layer(x)
            delta_w = layer._compute_weight_delta_simple(adapter)
            out_merged = x @ (base.weight.data + delta_w).t()

        assert torch.allclose(out_adapter, out_merged, atol=1e-4), (
            f"Max diff: {(out_adapter - out_merged).abs().max().item()}"
        )

    def test_merge_equals_forward_nonzero_theta(self):
        """Merge is exact even when theta (rotation) parameters are nonzero."""
        from peft.tuners.jora.layer import JoraLayer

        torch.manual_seed(7)
        d = 16
        k = 4
        base = nn.Linear(d, d, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=k,
            S_L=k * 2,
            S_R=k * 2,
            selection="none",
            rotation_impl="torch",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]

        with torch.no_grad():
            # Set support to first k*2 indices
            idxs = torch.arange(k * 2, dtype=torch.long)
            adapter.core.set_support(idxs)
            adapter.core.delta.uniform_(-0.5, 0.5)
            # Set nonzero rotations
            adapter.theta_L.uniform_(-0.3, 0.3)
            adapter.theta_R.uniform_(-0.3, 0.3)
            # Initialize pairs with k valid pairs (consecutive: (0,1), (2,3), ...)
            pairs = torch.arange(k * 2, dtype=torch.long).view(k, 2)
            adapter.pairs_L[:k].copy_(pairs)
            adapter.pairs_L[k:].fill_(-1)
            adapter.pairs_R[:k].copy_(pairs)
            adapter.pairs_R[k:].fill_(-1)
            adapter.num_pairs_L.fill_(k)
            adapter.num_pairs_R.fill_(k)
            # Sync Python-side counter cache
            adapter._num_pairs_py_initialized = True
            adapter._num_pairs_py = {'left': k, 'right': k}

        x = torch.randn(8, d)
        with torch.no_grad():
            out_adapter = layer(x)
            delta_w = layer._compute_weight_delta_simple(adapter)
            out_merged = x @ (base.weight.data + delta_w).t()

        max_diff = (out_adapter - out_merged).abs().max().item()
        assert max_diff < 1e-3, f"Merge not exact with nonzero theta: max_diff={max_diff}"


# ------------------------------------------- Gradient-live after support allocation ---

class TestGradientFlow:
    """After support allocation, both theta and delta should receive nonzero gradients."""

    def test_theta_and_delta_grads_after_support_set(self):
        from peft.tuners.jora.layer import JoraLayer

        torch.manual_seed(0)
        d = 16
        k = 4
        base = nn.Linear(d, d, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=k,
            S_L=k * 2,
            S_R=k * 2,
            selection="none",
            rotation_impl="torch",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]

        # Set support
        idxs = torch.arange(k * 2, dtype=torch.long)
        with torch.no_grad():
            adapter.core.set_support(idxs)
            adapter.core.delta.zero_()
            adapter.theta_L.zero_()
            adapter.theta_R.zero_()
            # Initialize pairs with k valid pairs (consecutive: (0,1), (2,3), ...)
            pairs = torch.arange(k * 2, dtype=torch.long).view(k, 2)
            adapter.pairs_L[:k].copy_(pairs)
            adapter.pairs_L[k:].fill_(-1)
            adapter.pairs_R[:k].copy_(pairs)
            adapter.pairs_R[k:].fill_(-1)
            adapter.num_pairs_L.fill_(k)
            adapter.num_pairs_R.fill_(k)
            # Sync Python-side counter cache
            adapter._num_pairs_py_initialized = True
            adapter._num_pairs_py = {'left': k, 'right': k}

        # Forward + backward
        x = torch.randn(2, d)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert adapter.core.delta.grad is not None, "delta has no gradient"
        assert adapter.core.delta.grad.abs().max().item() > 0, "delta gradient is all zeros"

        assert adapter.theta_L.grad is not None, "theta_L has no gradient"
        assert adapter.theta_L.grad.abs().max().item() > 0, "theta_L gradient is all zeros after support set"

        assert adapter.theta_R.grad is not None, "theta_R has no gradient"
        assert adapter.theta_R.grad.abs().max().item() > 0, "theta_R gradient is all zeros after support set"


# ------------------------------------------- Checkpoint/resume safety -------

class TestCheckpointResume:
    """After a state_dict round-trip, frozen support stays frozen and training-state buffers are preserved."""

    def _build_frozen_layer(self, d: int = 16, k: int = 4):
        from peft.tuners.jora.layer import JoraLayer

        base = nn.Linear(d, d, bias=False)
        cfg = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=k,
            S_L=k * 2,
            S_R=k * 2,
            selection="topk_ema",
            rotation_impl="torch",
        )
        layer = JoraLayer(base_layer=base, adapter_name="default", cfg=cfg)
        adapter = layer.adapters["default"]

        # Freeze it (simulates post-warmup state)
        adapter._freeze_support_if_needed()
        assert adapter._pairs_frozen is True
        assert bool(adapter.pairs_frozen_flag.item())
        return layer, adapter

    def test_frozen_flag_survives_state_dict_roundtrip(self):
        """After state_dict -> load_state_dict, _pairs_frozen must still be True."""
        layer, adapter = self._build_frozen_layer()

        # Capture state dict
        sd = layer.state_dict()

        # Build a fresh (unfrozen) layer
        d, k = 16, 4
        from peft.tuners.jora.layer import JoraLayer
        base2 = nn.Linear(d, d, bias=False)
        cfg2 = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=k,
            S_L=k * 2,
            S_R=k * 2,
            rotation_impl="torch",
        )
        layer2 = JoraLayer(base_layer=base2, adapter_name="default", cfg=cfg2)
        adapter2 = layer2.adapters["default"]

        assert not adapter2._pairs_frozen, "Expected fresh layer to be unfrozen"

        # Load the state dict from the frozen layer
        layer2.load_state_dict(sd)

        # The freeze gate must be restored
        assert bool(adapter2.pairs_frozen_flag.item()), "pairs_frozen_flag not restored after load_state_dict"
        assert adapter2._pairs_frozen is True, "_pairs_frozen not restored via _restore_frozen_flag hook"

    def test_training_state_buffers_in_state_dict(self):
        """grad_row_ema, grad_col_ema, step_idx, ema_step_idx must appear in state_dict."""
        layer, adapter = self._build_frozen_layer()
        sd = layer.state_dict()
        keys = set(sd.keys())
        required_suffixes = [
            "pairs_frozen_flag",
            "grad_row_ema",
            "grad_col_ema",
            "step_idx",
            "ema_step_idx",
        ]
        for suffix in required_suffixes:
            found = any(k.endswith(suffix) for k in keys)
            assert found, f"Buffer '{suffix}' missing from state_dict. Keys: {list(keys)}"

    def test_update_step_frozen_after_reload(self):
        """Resumed training must not mutate pairs after reload."""
        layer, adapter = self._build_frozen_layer()
        sd = layer.state_dict()

        d, k = 16, 4
        from peft.tuners.jora.layer import JoraLayer
        base2 = nn.Linear(d, d, bias=False)
        cfg2 = JoraConfig.paper_path(
            target_modules=["q_proj"],
            k=k,
            S_L=k * 2,
            S_R=k * 2,
            selection="topk_ema",
            rotation_impl="torch",
        )
        layer2 = JoraLayer(base_layer=base2, adapter_name="default", cfg=cfg2)
        layer2.load_state_dict(sd)
        adapter2 = layer2.adapters["default"]

        pairs_L_before = adapter2.pairs_L.clone()
        pairs_R_before = adapter2.pairs_R.clone()

        adapter2.update_step(50, 100)
        adapter2.update_step(99, 100)

        assert torch.equal(pairs_L_before, adapter2.pairs_L), "pairs_L mutated after reload+resume"
        assert torch.equal(pairs_R_before, adapter2.pairs_R), "pairs_R mutated after reload+resume"
