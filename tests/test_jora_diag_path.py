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
        """Merged weights should produce the same output as the adapter forward pass."""
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
