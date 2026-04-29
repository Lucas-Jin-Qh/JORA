# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for JORA-Diag path (DiagCore + rotation), diag_path(), and selective_path() factories."""

import pytest
import torch
import torch.nn as nn
from peft import JoraConfig
from peft.tuners.jora.layer import JoraLayer
from peft.tuners.jora.core import DiagCore, BlockCore, LowRankCore, build_core


# ------------------------------------------------------------------ factories ---

class TestDiagPathFactory:
    def test_diag_path_factory_defaults(self):
        cfg = JoraConfig.diag_path(target_modules=["q_proj"])
        assert cfg.core == "diag"
        assert cfg.theta_init_std == 2e-3
        assert cfg.core_init_std == 5e-3
        assert cfg.selection == "none"
        assert cfg.magnitude == "none"
        assert cfg.zero_init_core is True
        assert cfg.S_L == 32
        assert cfg.S_R == 32

    def test_diag_path_override(self):
        cfg = JoraConfig.diag_path(target_modules=["q_proj"], S_L=16, S_R=16, theta_init_std=5e-3)
        assert cfg.core == "diag"
        assert cfg.S_L == 16
        assert cfg.S_R == 16
        assert cfg.theta_init_std == 5e-3  # overridden

    def test_selective_path_factory_defaults(self):
        cfg = JoraConfig.selective_path(target_modules=["q_proj"])
        assert cfg.core == "selective_diag"
        assert cfg.selection == "topk_ema"
        assert cfg.theta_init_std == 0.0
        assert cfg.pairs_freeze_after_warmup is True

    def test_selective_path_override(self):
        cfg = JoraConfig.selective_path(k=16, warmup_steps=100)
        assert cfg.core == "selective_diag"
        assert cfg.k == 16
        assert cfg.warmup_steps == 100

    def test_paper_path_unchanged(self):
        """Ensure paper_path() still returns selective_diag defaults."""
        cfg = JoraConfig.paper_path(target_modules=["q_proj"])
        assert cfg.core == "selective_diag"
        assert cfg.magnitude == "none"
        assert cfg.zero_init_core is True
        assert cfg.theta_init_std == 0.0
        assert cfg.pairs_freeze_after_warmup is True


# ------------------------------------------------------------------ init tests ---

class TestDiagCoreInit:
    def test_zero_init_delta_near_zero(self):
        """zero_init_core=True should give ||delta||/||x|| ≈ 0."""
        base = nn.Linear(64, 64, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=True)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(4, 64)
        with torch.no_grad():
            delta = layer.adapters["default"].compute_delta(x)
        ratio = delta.norm() / x.norm()
        assert ratio < 1e-3, f"Zero-init delta ratio too large: {ratio}"

    def test_nonzero_init_small_effect(self):
        """nonzero init should give small but nonzero delta."""
        base = nn.Linear(64, 64, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.005)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(4, 64)
        with torch.no_grad():
            delta = layer.adapters["default"].compute_delta(x)
        ratio = delta.norm() / x.norm()
        assert 1e-5 < ratio < 1e-1, f"Nonzero init ratio out of range: {ratio}"

    def test_nonzero_theta_init(self):
        """theta should be nonzero for diag_path."""
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4)
        layer = JoraLayer(base, "default", cfg)
        adapter = layer.adapters["default"]
        assert adapter.theta_L is not None
        assert adapter.theta_R is not None
        assert adapter.theta_L.std().item() > 0, "theta_L should be nonzero for diag_path"
        assert adapter.theta_R.std().item() > 0, "theta_R should be nonzero for diag_path"

    def test_zero_init_core_still_gets_gradient(self):
        """zero-init core should still be trainable under backprop."""
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=True)
        layer = JoraLayer(base, "default", cfg)
        layer.train()

        adapter = layer.adapters["default"]
        assert torch.count_nonzero(adapter.core.diag_params).item() == 0

        x = torch.randn(2, 32)
        loss = layer(x).sum()
        loss.backward()

        assert adapter.core.diag_params.grad is not None
        assert torch.isfinite(adapter.core.diag_params.grad).all()
        assert adapter.core.diag_params.grad.abs().sum().item() > 0

    def test_nonzero_init_zero_std_still_zero(self):
        """zero_init_core=False but core_init_std=0.0 should give numerically zero core."""
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.0)
        layer = JoraLayer(base, "default", cfg)
        adapter = layer.adapters["default"]
        assert adapter.core.diag_params.abs().max().item() == 0.0
        assert torch.count_nonzero(adapter.core.diag_params).item() == 0

    def test_theta_zero_for_selective_path(self):
        """theta should be zero for selective_path."""
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.selective_path(S_L=4, S_R=4)
        layer = JoraLayer(base, "default", cfg)
        adapter = layer.adapters["default"]
        assert adapter.theta_L is not None
        assert adapter.theta_R is not None
        assert adapter.theta_L.abs().max().item() == 0.0, "theta_L should be zero for selective_path"
        assert adapter.theta_R.abs().max().item() == 0.0, "theta_R should be zero for selective_path"


# ------------------------------------------------------------------ param counts ---

class TestDiagCoreParamCount:
    def test_diag_core_param_count(self):
        core = DiagCore(n=4096, m=4096, device="cpu", dtype=torch.float32)
        assert core.diag_params.numel() == 4096

    def test_block_core_param_count(self):
        """BlockCore stores [n_blocks, block_size, block_size] dense blocks."""
        core = BlockCore(n=4096, m=4096, device="cpu", dtype=torch.float32, block_size=4)
        n_blocks = 4096 // 4  # = 1024
        assert core.blocks.shape == (n_blocks, 4, 4)
        assert core.blocks.numel() == n_blocks * 4 * 4

    def test_lowrank_core_param_count(self):
        core = LowRankCore(n=4096, m=4096, device="cpu", dtype=torch.float32, rank=8)
        assert core.A.numel() == 4096 * 8
        assert core.B.numel() == 4096 * 8


# ------------------------------------------------------------------ build_core with init_std ---

class TestBuildCoreWithInitStd:
    def test_build_diag_uses_init_std(self):
        cfg = JoraConfig(core_init_std=0.02)
        core = build_core("diag", 512, 512, device="cpu", dtype=torch.float32, cfg=cfg)
        assert abs(core.diag_params.std().item() - 0.02) < 0.005

    def test_build_block_uses_init_std(self):
        cfg = JoraConfig(core_init_std=0.02)
        core = build_core("block", 512, 512, device="cpu", dtype=torch.float32, cfg=cfg)
        assert abs(core.blocks.std().item() - 0.02) < 0.005

    def test_build_lowrank_uses_init_std(self):
        cfg = JoraConfig(core_init_std=0.02)
        core = build_core("lowrank", 512, 512, device="cpu", dtype=torch.float32, cfg=cfg)
        assert abs(core.A.std().item() - 0.02) < 0.005
        assert abs(core.B.std().item() - 0.02) < 0.005

    def test_build_diag_zero_std_nonzero_zero_init(self):
        """zero_init_core=False but core_init_std=0.0 should give numerically zero diag core."""
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.0)
        core = build_core("diag", 512, 512, device="cpu", dtype=torch.float32, cfg=cfg)
        assert core.diag_params.abs().max().item() == 0.0
        assert torch.count_nonzero(core.diag_params).item() == 0

    def test_build_block_zero_std_nonzero_zero_init(self):
        """zero_init_core=False but core_init_std=0.0 should give numerically zero block core."""
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.0)
        core = build_core("block", 512, 512, device="cpu", dtype=torch.float32, cfg=cfg)
        assert core.blocks.abs().max().item() == 0.0
        assert torch.count_nonzero(core.blocks).item() == 0

    def test_build_lowrank_zero_std_nonzero_zero_init(self):
        """zero_init_core=False but core_init_std=0.0 should give numerically zero lowrank core."""
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.0)
        core = build_core("lowrank", 512, 512, device="cpu", dtype=torch.float32, cfg=cfg)
        assert core.A.abs().max().item() == 0.0
        assert torch.count_nonzero(core.A).item() == 0
        assert core.B.abs().max().item() == 0.0
        assert torch.count_nonzero(core.B).item() == 0


# ------------------------------------------------------------------ merge consistency ---

class TestDiagCoreMerge:
    def test_diag_core_merge_equals_forward(self):
        """Merged weights should produce the same output as the adapter forward pass.

        C1.6 FIX: DiagCore now uses exact basis-probing merge (same as SelectiveDiagCore),
        so merged_output == adapter_forward even with nonzero theta and rotations.
        """
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=8, S_R=8)
        layer = JoraLayer(base, "default", cfg)
        adapter = layer.adapters["default"]

        x = torch.randn(4, 32)
        with torch.no_grad():
            out_adapter = layer(x)
            delta_w = layer._compute_weight_delta_simple(adapter)
            out_merged = x @ (base.weight.data + delta_w).t()

        torch.testing.assert_close(out_adapter, out_merged, atol=1e-4, rtol=1e-3)

    def test_diag_core_unmerge_equals_original(self):
        """Unmerged output should equal the original base output when adapter is identity."""
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=0, S_R=0)  # no rotation
        layer = JoraLayer(base, "default", cfg)

        x = torch.randn(4, 32)
        with torch.no_grad():
            out_base = base(x)
            layer.merge()
            out_merged = layer(x)
            layer.unmerge()
            out_unmerged = layer(x)

        torch.testing.assert_close(out_unmerged, out_base, atol=1e-4, rtol=1e-3)


# ------------------------------------------------------------------ full forward pass ---

class TestDiagCoreForward:
    def test_forward_no_nan(self):
        base = nn.Linear(64, 64, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(2, 64)
        out = layer(x)
        assert torch.isfinite(out).all(), "Output has NaN/Inf"

    def test_training_mode_backward(self):
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4)
        layer = JoraLayer(base, "default", cfg)
        layer.train()

        x = torch.randn(2, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        adapter = layer.adapters["default"]
        assert adapter.theta_L.grad is not None
        assert adapter.theta_R.grad is not None
        assert adapter.core.diag_params.grad is not None
        assert torch.isfinite(adapter.theta_L.grad).all()
        assert torch.isfinite(adapter.core.diag_params.grad).all()


# ------------------------------------------------------------------ C1.5 residualized tests ---

class TestResidualizedDiagCore:
    """DiagCore backward-compatibility regression suite.

    After the Option C failure (2026-04-28), DiagCore was reverted to additive form.
    These tests verify the reverted additive behavior:
        - apply_to_vector returns Diag(d) @ x  (not (I + Diag(d)) @ x)
        - compute_delta returns R_L^T @ Diag(d) @ R_R @ x  (not residualized form)
        - merge uses exact basis-probing (C1.6 fix, preserved after revert)
    """

    def test_zero_init_theta_zero_delta_zero(self):
        """G1: d=0, theta=0 → Δ(x) ≈ 0 (strict zero function change)."""
        base = nn.Linear(64, 64, bias=False)
        # selective_path uses theta=0 at init, which is equivalent to identity rotation
        cfg = JoraConfig.selective_path(S_L=4, S_R=4, k=4)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(4, 64)
        with torch.no_grad():
            delta = layer.adapters["default"].compute_delta(x)
        ratio = delta.norm() / x.norm()
        assert ratio < 1e-6, f"G1 FAIL: delta/x ratio {ratio:.2e} > 1e-6"

    def test_nonzero_theta_nearzero_delta(self):
        """G2: theta~N(0,2e-3), d=0 → |Δ|/|x| < 1e-3 (near-zero)."""
        base = nn.Linear(64, 64, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=True)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(4, 64)
        with torch.no_grad():
            delta = layer.adapters["default"].compute_delta(x)
        ratio = delta.norm() / x.norm()
        assert ratio < 1e-3, f"G2 FAIL: delta/x ratio {ratio:.2e} > 1e-3"

    def test_nonzero_d_bounded_delta(self):
        """G3: nonzero d init → Δ bounded in [1e-5, 1e-1]."""
        base = nn.Linear(64, 64, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.005)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(4, 64)
        with torch.no_grad():
            delta = layer.adapters["default"].compute_delta(x)
        ratio = delta.norm() / x.norm()
        assert 1e-5 < ratio < 1e-1, f"G3 FAIL: delta/x ratio {ratio:.2e} outside [1e-5, 1e-1]"

    def test_norot_degenerates_to_diag_dx(self):
        """G6: NoRot (S_L=0, S_R=0) → Δ(x) = Diag(d) x (identity core becomes Diag(d))."""
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=0, S_R=0, zero_init_core=False, core_init_std=0.01)
        layer = JoraLayer(base, "default", cfg)
        adapter = layer.adapters["default"]

        x = torch.randn(4, 32)
        with torch.no_grad():
            delta = adapter.compute_delta(x)
            expected = x * adapter.core.diag_params

        torch.testing.assert_close(delta, expected, atol=1e-5, rtol=1e-4)

    def test_apply_to_vector_additive_shape(self):
        """apply_to_vector with n > m: learned range + zeros (additive, no identity)."""
        from peft.tuners.jora.core import DiagCore
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            core = DiagCore(n=8, m=4, device="cpu", dtype=torch.float32, zero_init=False, init_std=0.01)

        x = torch.randn(3, 4)
        y = core.apply_to_vector(x)
        assert y.shape == (3, 8), f"Expected shape (3, 8), got {y.shape}"

        # Additive: y[:4] = d * x (small), y[4:] = 0
        # With d~N(0, 0.01), the contribution is tiny
        assert y[..., 4:].abs().max().item() < 1e-6, f"Pad region should be zeros: {y[..., 4:]}"
        # The learned range is nonzero (d != 0)
        assert y[..., :4].abs().max().item() > 0, "Learned range should be nonzero with nonzero d"

    def test_apply_to_vector_additive_nonzero_d(self):
        """apply_to_vector: d != 0 should return d * x (additive, not (1+d) * x)."""
        from peft.tuners.jora.core import DiagCore
        core = DiagCore(n=4, m=4, device="cpu", dtype=torch.float32, zero_init=False, init_std=0.1)

        x = torch.ones(1, 4)
        y = core.apply_to_vector(x)
        # Expected: y = d * x = d (additive form)
        expected = core.diag_params
        torch.testing.assert_close(y.squeeze(0), expected, atol=1e-5, rtol=1e-4)

    def test_forward_deprecated_returns_core_matrix(self):
        """Deprecated forward() returns I + Diag(d) (the full core matrix)."""
        from peft.tuners.jora.core import DiagCore
        import warnings
        core = DiagCore(n=4, m=4, device="cpu", dtype=torch.float32, zero_init=False, init_std=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = core.forward()
        expected_diag = 1.0 + core.diag_params
        diag_from_D = torch.diag(D)
        torch.testing.assert_close(diag_from_D, expected_diag, atol=1e-5, rtol=1e-4)

    def test_get_row_slice_core_matrix(self):
        """get_row_slice: row i should have D[i,i] = 1 + d[i]."""
        from peft.tuners.jora.core import DiagCore
        import warnings
        core = DiagCore(n=4, m=4, device="cpu", dtype=torch.float32, zero_init=False, init_std=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = core.forward()
        for i in range(4):
            sl = core.get_row_slice(i, i + 1)
            diag_val = sl[0, i].item()
            expected = (1.0 + core.diag_params[i]).item()
            assert abs(diag_val - expected) < 1e-5, f"Row {i}: got {diag_val}, expected {expected}"

    def test_backward_gradient_liveness(self):
        """G4: backward produces nonzero gradients for both d and theta."""
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.005)
        layer = JoraLayer(base, "default", cfg)
        layer.train()

        adapter = layer.adapters["default"]
        x = torch.randn(2, 32)
        loss = layer(x).sum()
        loss.backward()

        assert adapter.core.diag_params.grad is not None
        assert adapter.core.diag_params.grad.abs().sum().item() > 0, "G4 FAIL: d has no gradient"
        assert adapter.theta_L.grad is not None
        assert adapter.theta_L.grad.abs().sum().item() > 0, "G4 FAIL: theta_L has no gradient"

    def test_full_forward_no_nan(self):
        """Residualized DiagCore produces finite output."""
        base = nn.Linear(64, 64, bias=False)
        cfg = JoraConfig.diag_path(S_L=4, S_R=4, zero_init_core=False, core_init_std=0.005)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(2, 64)
        out = layer(x)
        assert torch.isfinite(out).all(), "Output has NaN/Inf"

    def test_selective_still_works(self):
        """G7 regression: SelectiveDiagCore must be unaffected by DiagCore changes."""
        from peft.tuners.jora.core import SelectiveDiagCore
        base = nn.Linear(32, 32, bias=False)
        cfg = JoraConfig.selective_path(S_L=4, S_R=4, k=4)
        layer = JoraLayer(base, "default", cfg)
        x = torch.randn(2, 32)
        out = layer(x)
        assert torch.isfinite(out).all(), "Selective output has NaN/Inf"
        loss = out.sum()
        loss.backward()
        adapter = layer.adapters["default"]
        if adapter.core.delta.grad is not None:
            assert torch.isfinite(adapter.core.delta.grad).all(), "Selective delta grad has NaN/Inf"

    def test_blockcore_still_works(self):
        """G8 regression: BlockCore must be unaffected by DiagCore changes."""
        from peft.tuners.jora.core import BlockCore
        core = BlockCore(n=4, m=4, device="cpu", dtype=torch.float32, block_size=2, zero_init=False, init_std=0.01)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = core.forward()
        x = torch.randn(2, 4)
        y = core.apply_to_vector(x)
        assert y.shape == (2, 4)
        assert torch.isfinite(y).all(), "BlockCore output has NaN/Inf"
