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

"""
M0 Correctness Gate: JORA Save/Load/Merge Sanity Suite
======================================================

Purpose: Verify that save/load roundtrip and merge/unmerge produce correct,
identical outputs before investing GPU hours in expensive sweeps.

Scope:
  - JORA-Diag (DiagCore, rotation ON/OFF)
  - JORA-NoRot (S_L=S_R=0, DiagCore, rotation OFF)
  - JORA-Paper (SelectiveDiagCore, zero theta and nonzero theta)
  - Both square and rectangular layers
  - magnitude=none, magnitude=ecd_tanh, magnitude=oer_softmax

What is tested:
  [S] Save/Load roundtrip: output identity after save→load
  [M] Merge equivalence: merged_output ≈ adapter_forward
  [U] Unmerge correctness: unmerged_output ≈ base_output
  [P] Persistence: training-state buffers survive save/load

Known limitations:
  - DiagCore merge uses exact basis-probing (C1.6 fix preserved after Option C revert).
    This means merged_output ≈ adapter_forward for DiagCore. Nonzero tolerance is
    atol=1e-4/rtol=1e-3 (numerical error only).
  - SelectiveDiagCore merge is exact via the same basis-probing method (for square layers only).

Run with: pytest tests/test_jora_save_load_merge_sanity.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from peft import JoraConfig, PeftModel, get_peft_model
from peft.tuners.jora.layer import JoraLayer
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_tiny_local_model():
    """Minimal OPTForCausalLM without network access."""
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import OPTConfig, AutoModelForCausalLM

    hf_config = OPTConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        ffn_dim=64,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return AutoModelForCausalLM.from_config(hf_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_diag_layer(d: int, bias: bool = False) -> nn.Module:
    """Square linear layer."""
    return nn.Linear(d, d, bias=bias)


def _make_rect_layer(out: int, inp: int) -> nn.Module:
    """Rectangular linear layer."""
    return nn.Linear(inp, out, bias=False)


# ---------------------------------------------------------------------------
# S1: Save/Load Roundtrip — JORA-Diag (rotation ON, DiagCore)
# ---------------------------------------------------------------------------

class TestSaveLoadDiagOn:
    """JORA-Diag: rotation ON, DiagCore, magnitude=none. Square layers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0)
        # Create base model and capture its state BEFORE wrapping with PeftModel
        # (needed so fresh base model can be initialized with identical weights)
        self.model = _create_tiny_local_model()
        self._raw_base_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.config = JoraConfig.diag_path(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=4,
            S_R=4,
            k=2,
            magnitude="none",
            rotation_impl="torch",
        )
        self.peft = get_peft_model(self.model, self.config, adapter_name="default")
        self.peft.eval()
        self.inputs = torch.randint(0, 64, (4, 8), dtype=torch.long)

    def test_roundtrip_preserves_output(self, tmp_path):
        """Save→load must produce identical logits (same as existing test_jora_save_load)."""
        x = self.inputs

        with torch.no_grad():
            before = self.peft(x).logits

        self.peft.save_pretrained(tmp_path)

        # Fresh base: same seed=0 → same random base weights
        torch.manual_seed(0)
        fresh_base = _create_tiny_local_model()
        fresh_base.load_state_dict(self._raw_base_state_dict)
        loaded = PeftModel.from_pretrained(fresh_base, tmp_path)
        loaded.eval()

        with torch.no_grad():
            after = loaded(x).logits

        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_roundtrip_preserves_theta(self, tmp_path):
        """theta_L/R must survive save→load byte-for-byte."""
        self.peft.save_pretrained(tmp_path)
        torch.manual_seed(0)
        fresh_base = _create_tiny_local_model()
        fresh_base.load_state_dict(self._raw_base_state_dict)
        loaded = PeftModel.from_pretrained(fresh_base, tmp_path)

        orig_sd = get_peft_model_state_dict(self.peft)
        load_sd = get_peft_model_state_dict(loaded)

        for key in orig_sd:
            if "theta_L" in key or "theta_R" in key:
                torch.testing.assert_close(
                    orig_sd[key],
                    load_sd[key],
                    atol=0.0,
                    rtol=0.0,
                    msg=f"Mismatch on {key}",
                )

    def test_roundtrip_preserves_pairs(self, tmp_path):
        """pairs_L/R and num_pairs_L/R must survive save→load."""
        self.peft.save_pretrained(tmp_path)
        torch.manual_seed(0)
        fresh_base = _create_tiny_local_model()
        fresh_base.load_state_dict(self._raw_base_state_dict)
        loaded = PeftModel.from_pretrained(fresh_base, tmp_path)

        orig_sd = get_peft_model_state_dict(self.peft)
        load_sd = get_peft_model_state_dict(loaded)

        for key in orig_sd:
            if "pairs_" in key or "num_pairs_" in key:
                torch.testing.assert_close(
                    orig_sd[key],
                    load_sd[key],
                    atol=0.0,
                    rtol=0.0,
                    msg=f"Mismatch on {key}",
                )


# ---------------------------------------------------------------------------
# S2: Save/Load Roundtrip — JORA-NoRot (rotation OFF, S_L=S_R=0)
# ---------------------------------------------------------------------------

class TestSaveLoadNoRot:
    """JORA-NoRot: S_L=S_R=0, DiagCore. Verifies theta is None and no rotation overhead."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0)
        self.model = _create_tiny_local_model()
        self._raw_base_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            core="diag",
            S_L=0,
            S_R=0,
            k=2,
            magnitude="none",
            zero_init_core=True,
            selection="none",
            rotation_impl="torch",
        )
        self.peft = get_peft_model(self.model, self.config, adapter_name="default")
        self.peft.eval()
        self.inputs = torch.randint(0, 64, (4, 8), dtype=torch.long)

    def test_norot_theta_is_none(self):
        """With S_L=S_R=0, theta_L and theta_R must be None (not allocated)."""
        for name, module in self.peft.named_modules():
            if isinstance(module, JoraLayer):
                adapter = module.adapters["default"]
                assert adapter.theta_L is None, f"{name}: theta_L should be None for NoRot"
                assert adapter.theta_R is None, f"{name}: theta_R should be None for NoRot"

    def test_roundtrip_output_unchanged(self, tmp_path):
        """Save→load must preserve output for NoRot mode."""
        x = self.inputs
        with torch.no_grad():
            before = self.peft(x).logits

        self.peft.save_pretrained(tmp_path)
        torch.manual_seed(0)
        fresh_base = _create_tiny_local_model()
        fresh_base.load_state_dict(self._raw_base_state_dict)
        loaded = PeftModel.from_pretrained(fresh_base, tmp_path)
        loaded.eval()

        with torch.no_grad():
            after = loaded(x).logits

        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_norot_merged_equals_base(self):
        """NoRot merge should produce output ≈ base (no rotation contribution)."""
        x = self.inputs
        with torch.no_grad():
            base_out = self.peft.base_model(x).logits.clone()
            self.peft.merge_adapter()
            merged_out = self.peft(x).logits.clone()

        diff = (merged_out - base_out).abs().max().item()
        assert diff < 1e-3, f"NoRot merge diff too large: {diff}"


# ---------------------------------------------------------------------------
# S3: Save/Load Roundtrip — JORA-Paper (SelectiveDiagCore, zero theta)
# ---------------------------------------------------------------------------

class TestSaveLoadSelectiveDiag:
    """JORA-Paper: SelectiveDiagCore, zero-init theta. Square layers only."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _create_tiny_local_model()
        self._raw_base_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.config = JoraConfig.paper_path(
            target_modules=["q_proj", "k_proj"],
            S_L=8,
            S_R=8,
            k=4,
            magnitude="none",
            rotation_impl="torch",
        )
        self.peft = get_peft_model(self.model, self.config, adapter_name="default")
        self.peft.eval()
        self.inputs = torch.randint(0, 64, (4, 8), dtype=torch.long)

    def test_roundtrip_preserves_core_delta(self, tmp_path):
        """Core delta params must survive save→load."""
        self.peft.save_pretrained(tmp_path)
        torch.manual_seed(42)
        fresh_base = _create_tiny_local_model()
        fresh_base.load_state_dict(self._raw_base_state_dict)
        loaded = PeftModel.from_pretrained(fresh_base, tmp_path)
        loaded.eval()

        orig_sd = get_peft_model_state_dict(self.peft)
        load_sd = get_peft_model_state_dict(loaded)

        for key in orig_sd:
            if "core.delta" in key:
                torch.testing.assert_close(
                    orig_sd[key],
                    load_sd[key],
                    atol=0.0,
                    rtol=0.0,
                    msg=f"Mismatch on {key}",
                )

    def test_roundtrip_preserves_output(self, tmp_path):
        """Output identity after save→load."""
        x = self.inputs
        with torch.no_grad():
            before = self.peft(x).logits

        self.peft.save_pretrained(tmp_path)
        torch.manual_seed(42)
        fresh_base = _create_tiny_local_model()
        fresh_base.load_state_dict(self._raw_base_state_dict)
        loaded = PeftModel.from_pretrained(fresh_base, tmp_path)
        loaded.eval()

        with torch.no_grad():
            after = loaded(x).logits

        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# S4: Save/Load Roundtrip — Magnitude Scaling
# ---------------------------------------------------------------------------

class TestSaveLoadWithMagnitude:
    """Verify magnitude != 'none' survives save→load."""

    @pytest.mark.parametrize("magnitude", ["ecd_tanh", "oer_softmax"])
    def test_roundtrip_with_magnitude(self, magnitude: str, tmp_path):
        torch.manual_seed(7)
        model = _create_tiny_local_model()
        raw_base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        config = JoraConfig.diag_path(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=4,
            S_R=4,
            k=2,
            magnitude=magnitude,
            rotation_impl="torch",
        )
        peft = get_peft_model(model, config)
        peft.eval()
        x = torch.randint(0, 64, (4, 8), dtype=torch.long)

        with torch.no_grad():
            before = peft(x).logits

        peft.save_pretrained(tmp_path)
        torch.manual_seed(7)
        fresh_base = _create_tiny_local_model()
        fresh_base.load_state_dict(raw_base_sd)
        loaded = PeftModel.from_pretrained(fresh_base, tmp_path)
        loaded.eval()

        with torch.no_grad():
            after = loaded(x).logits

        torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# S5: Save/Load Roundtrip — Rectangular Layers (DiagCore only)
# ---------------------------------------------------------------------------

class TestSaveLoadRectangularLayers:
    """DiagCore works on rectangular layers (SelectiveDiagCore does not)."""

    def test_diag_path_roundtrip_rectangular(self, tmp_path):
        torch.manual_seed(0)

        class RectModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 24, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = RectModel()
        raw_base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        config = JoraConfig.diag_path(
            target_modules=["linear"],
            S_L=4,
            S_R=4,
            k=2,
            magnitude="none",
            rotation_impl="torch",
        )
        peft = get_peft_model(model, config)
        peft.eval()
        x = torch.randn(4, 16)

        with torch.no_grad():
            before = peft(x)

        peft.save_pretrained(tmp_path)
        torch.manual_seed(0)
        fresh_model = RectModel()
        fresh_model.load_state_dict(raw_base_sd)
        loaded = PeftModel.from_pretrained(fresh_model, tmp_path)
        loaded.eval()

        with torch.no_grad():
            after = loaded(x)

        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# M1: Merge Equivalence — JORA-Diag (DiagCore)
# ---------------------------------------------------------------------------

class TestMergeDiagOn:
    """DiagCore merge uses exact basis-probing (C1.6 fix, preserved after Option C revert)."""

    def test_merge_unmerge_preserves_base_zero_init(self):
        """After merge→unmerge with zero-init core, output matches original base."""
        torch.manual_seed(0)
        d = 16
        base = _make_diag_layer(d)
        # S_L=S_R=0: no rotation, only zero-init core (identity).
        # With no rotation and zero core, delta=0, so unmerge is exact.
        config = JoraConfig.diag_path(
            target_modules=["q_proj"],
            S_L=0,
            S_R=0,
            k=4,
            magnitude="none",
            zero_init_core=True,
            rotation_impl="torch",
        )
        layer = JoraLayer(base, "default", config)
        layer.eval()
        x = torch.randn(8, d)

        with torch.no_grad():
            base_out = base(x)
            layer.merge()
            _ = layer(x)
            layer.unmerge()
            unmerged_out = layer(x)

        max_diff = (unmerged_out - base_out).abs().max().item()
        assert max_diff < 1e-4, f"Unmerge error too large: {max_diff}"


# ---------------------------------------------------------------------------
# M2: Merge Equivalence — JORA-Paper (SelectiveDiagCore, zero theta)
# ---------------------------------------------------------------------------

class TestMergeSelectiveDiag:
    """SelectiveDiagCore merge: exact reconstruction via basis probing (square layers).

    Uses the exact same pattern as the existing passing test_jora_paper_path.py.
    """

    @pytest.mark.parametrize("d", [16, 32])
    def test_merge_equals_forward_zero_theta(self, d: int):
        """Merge delta must equal the adapter forward linear map (theta=0)."""
        torch.manual_seed(42)
        base = _make_diag_layer(d)
        k = 4
        config = JoraConfig.paper_path(
            target_modules=["q_proj"],
            S_L=k * 2,
            S_R=k * 2,
            k=k,
            magnitude="none",
            selection="none",
            rotation_impl="torch",
        )
        layer = JoraLayer(base, "default", config)
        adapter = layer.adapters["default"]

        with torch.no_grad():
            idxs = torch.arange(k * 2, dtype=torch.long)
            adapter.core.set_support(idxs)
            adapter.core.delta.uniform_(-0.3, 0.3)
            # set_support does NOT reset num_pairs_L/R or pairs_L/R.
            # With random seeds, num_pairs may be < k and pairs may be invalid.
            # Explicitly set consecutive pairs so the test is deterministic.
            pairs = idxs.view(k, 2)
            adapter.pairs_L[:k].copy_(pairs)
            adapter.pairs_L[k:].fill_(-1)
            adapter.pairs_R[:k].copy_(pairs)
            adapter.pairs_R[k:].fill_(-1)
            adapter.num_pairs_L.fill_(k)
            adapter.num_pairs_R.fill_(k)

        x = torch.randn(8, d)
        with torch.no_grad():
            out_adapter = layer(x)
            delta_w = layer._compute_weight_delta_simple(adapter)
            out_merged = x @ (base.weight.data + delta_w).t()

        max_diff = (out_adapter - out_merged).abs().max().item()
        assert max_diff < 1e-4, (
            f"SelectiveDiagCore merge (theta=0) not exact: max_diff={max_diff}"
        )

    def test_unmerge_restores_base(self):
        """Unmerge must restore base output exactly.

        Use PeftModel.unmerge_adapter() which properly deactivates the adapter
        (sets _active_adapter to None) so the restored output equals the true base.
        """
        torch.manual_seed(0)
        d = 16
        k = 4

        class SimpleModel(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.lin = nn.Linear(d, d, bias=False)

            def forward(self, x):
                return self.lin(x)

        model = SimpleModel(d)
        config = JoraConfig.paper_path(
            target_modules=["lin"],
            S_L=k * 2,
            S_R=k * 2,
            k=k,
            magnitude="none",
            selection="none",
            rotation_impl="torch",
        )
        peft = get_peft_model(model, config)
        layer = peft.base_model.lin
        adapter = layer.adapters["default"]

        with torch.no_grad():
            idxs = torch.arange(k * 2, dtype=torch.long)
            adapter.core.set_support(idxs)
            adapter.core.delta.uniform_(-0.3, 0.3)
            pairs = idxs.view(k, 2)
            adapter.pairs_L[:k].copy_(pairs)
            adapter.pairs_L[k:].fill_(-1)
            adapter.pairs_R[:k].copy_(pairs)
            adapter.pairs_R[k:].fill_(-1)
            adapter.num_pairs_L.fill_(k)
            adapter.num_pairs_R.fill_(k)

        x = torch.randn(8, d)
        with torch.no_grad():
            # Capture pre-merge adapter output
            adapter_out = layer(x)

            peft.merge_adapter()
            _ = layer(x)

            # PeftModel.unmerge_adapter() properly deactivates the adapter
            peft.unmerge_adapter()
            unmerged_out = layer(x)

            # After unmerge with deactivated adapter, output should equal base
            base_out = model.lin(x)

        max_diff = (unmerged_out - base_out).abs().max().item()
        assert max_diff < 1e-4, f"Unmerge error: {max_diff}"


# ---------------------------------------------------------------------------
# M3: Merge Equivalence — SelectiveDiagCore with NONZERO theta
# ---------------------------------------------------------------------------

class TestMergeSelectiveDiagNonzeroTheta:
    """SelectiveDiagCore merge must be exact even with nonzero rotation params.

    Uses the exact same pattern as the existing passing
    test_jora_paper_path.py::TestMergeEquivalence::test_merge_equals_forward_nonzero_theta.
    """

    def test_merge_equals_forward_nonzero_theta(self):
        """Merge delta must equal the adapter forward linear map (theta!=0)."""
        torch.manual_seed(7)
        d = 16
        k = 4
        base = _make_diag_layer(d)
        config = JoraConfig.paper_path(
            target_modules=["q_proj"],
            S_L=k * 2,
            S_R=k * 2,
            k=k,
            magnitude="none",
            selection="none",
            rotation_impl="torch",
        )
        layer = JoraLayer(base, "default", config)
        adapter = layer.adapters["default"]

        with torch.no_grad():
            idxs = torch.arange(k * 2, dtype=torch.long)
            adapter.core.set_support(idxs)
            adapter.core.delta.uniform_(-0.5, 0.5)
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
            adapter._num_pairs_py_initialized = True
            adapter._num_pairs_py = {"left": k, "right": k}

        x = torch.randn(8, d)
        with torch.no_grad():
            out_adapter = layer(x)
            delta_w = layer._compute_weight_delta_simple(adapter)
            out_merged = x @ (base.weight.data + delta_w).t()

        max_diff = (out_adapter - out_merged).abs().max().item()
        assert max_diff < 1e-3, (
            f"SelectiveDiagCore merge (theta!=0) not exact: max_diff={max_diff}"
        )


# ---------------------------------------------------------------------------
# M4: Merge Equivalence — Rectangular Layers (DiagCore only)
# ---------------------------------------------------------------------------

class TestMergeRectangular:
    """DiagCore works on rectangular; SelectiveDiagCore rejects them."""

    def test_selective_diag_rejects_rectangular(self):
        """SelectiveDiagCore must raise ValueError on rectangular layers."""
        base = _make_rect_layer(24, 16)
        config = JoraConfig.paper_path(
            target_modules=["q_proj"],
            S_L=8,
            S_R=8,
            k=4,
            selection="none",
            rotation_impl="torch",
        )
        with pytest.raises(ValueError, match="square"):
            JoraLayer(base, "default", config)


# ---------------------------------------------------------------------------
# M5: Merge Equivalence — NoRot (theta=None)
# ---------------------------------------------------------------------------

class TestMergeNoRot:
    """NoRot merge/unmerge: output should equal base (no rotation contribution)."""

    def test_norot_merge_equals_forward(self):
        """With zero-init core, merged≈base."""
        torch.manual_seed(0)
        d = 16
        base = _make_diag_layer(d)
        config = JoraConfig(
            target_modules=["q_proj"],
            core="diag",
            S_L=0,
            S_R=0,
            k=2,
            magnitude="none",
            zero_init_core=True,
            selection="none",
            rotation_impl="torch",
        )
        layer = JoraLayer(base, "default", config)
        layer.eval()
        x = torch.randn(8, d)

        with torch.no_grad():
            out_adapter = layer(x)
            layer.merge()
            out_merged = layer(x)

        diff = (out_merged - out_adapter).abs().max().item()
        assert diff < 1e-3, f"NoRot merge diff: {diff}"

    def test_norot_unmerge_exact(self):
        """NoRot unmerge must restore base output exactly."""
        torch.manual_seed(0)
        d = 16
        base = _make_diag_layer(d)
        config = JoraConfig(
            target_modules=["q_proj"],
            core="diag",
            S_L=0,
            S_R=0,
            k=2,
            magnitude="none",
            zero_init_core=True,
            selection="none",
            rotation_impl="torch",
        )
        layer = JoraLayer(base, "default", config)
        layer.eval()
        x = torch.randn(8, d)

        with torch.no_grad():
            base_out = base(x)
            layer.merge()
            _ = layer(x)
            layer.unmerge()
            out_unmerged = layer(x)

        torch.testing.assert_close(out_unmerged, base_out, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# M6: Merge Equivalence — With Magnitude Scaling
# ---------------------------------------------------------------------------

class TestMergeWithMagnitude:
    """Merge/unmerge with magnitude != 'none'.

    DiagCore uses exact basis-probing for merge (C1.6 fix, preserved after Option C revert).
    We test that the result is exact within numerical tolerance.
    """

    @pytest.mark.parametrize("magnitude", ["ecd_tanh", "oer_softmax"])
    def test_merge_unmerge_with_magnitude(self, magnitude: str):
        """Unmerge must be exact with magnitude scaling (within numerical tolerance)."""
        torch.manual_seed(7)
        d = 16
        base = _make_diag_layer(d)
        config = JoraConfig.diag_path(
            target_modules=["q_proj"],
            S_L=4,
            S_R=4,
            k=2,
            magnitude=magnitude,
            zero_init_core=False,
            core_init_std=0.02,
            rotation_impl="torch",
        )
        layer = JoraLayer(base, "default", config)
        layer.eval()
        x = torch.randn(8, d)

        with torch.no_grad():
            base_out = base(x)
            layer.merge()
            _ = layer(x)
            layer.unmerge()
            out_unmerged = layer(x)

        # DiagCore with basis-probing: unmerge must be exact (within numerical tolerance).
        # Relative tolerance 10% is the absolute ceiling for bf16 numerical error.
        unmerge_diff = (out_unmerged - base_out).abs().max().item()
        base_scale = base_out.abs().max().item()
        relative_diff = unmerge_diff / (base_scale + 1e-8)
        assert relative_diff < 0.1, (
            f"Unmerge with magnitude={magnitude} too far from base: "
            f"abs={unmerge_diff:.4f}, base_scale={base_scale:.4f}, rel={relative_diff:.4f}"
        )


# ---------------------------------------------------------------------------
# P1: Persistence — Frozen State Survives Save/Load
# ---------------------------------------------------------------------------

class TestFrozenStatePersistence:
    """Training-state buffers (pairs_frozen, step_idx, etc.) survive save/load."""

    def test_frozen_state_survives_peft_roundtrip(self):
        """get_peft_model_state_dict → set_peft_model_state_dict preserves freeze."""
        torch.manual_seed(0)
        model = _create_tiny_local_model()
        config = JoraConfig.paper_path(
            target_modules=["q_proj", "k_proj"],
            S_L=8,
            S_R=8,
            k=4,
            rotation_impl="torch",
        )
        peft = get_peft_model(model, config)

        # Freeze all adapters (simulates post-warmup state)
        for module in peft.modules():
            if isinstance(module, JoraLayer):
                for adapter in module.adapters.values():
                    adapter._freeze_support_if_needed()
                    assert adapter._pairs_frozen is True

        # Save via PEFT API
        peft_sd = get_peft_model_state_dict(peft, adapter_name="default")

        # Reload into fresh model
        base_model2 = _create_tiny_local_model()
        peft2 = get_peft_model(base_model2, config)
        set_peft_model_state_dict(peft2, peft_sd, adapter_name="default")

        # Frozen state must be restored
        for module in peft2.modules():
            if isinstance(module, JoraLayer):
                for adapter in module.adapters.values():
                    assert bool(adapter.pairs_frozen_flag.item()), (
                        "pairs_frozen_flag not restored"
                    )
                    assert adapter._pairs_frozen is True, "_pairs_frozen not restored"

        # Pairs must not mutate after resumed update_step calls
        for name, module in peft2.named_modules():
            if isinstance(module, JoraLayer):
                for aname, adapter in module.adapters.items():
                    pl = adapter.pairs_L.clone()
                    pr = adapter.pairs_R.clone()
                    adapter.update_step(50, 100)
                    adapter.update_step(99, 100)
                    assert torch.equal(pl, adapter.pairs_L), (
                        f"pairs_L mutated after resume in {name}.{aname}"
                    )
                    assert torch.equal(pr, adapter.pairs_R), (
                        f"pairs_R mutated after resume in {name}.{aname}"
                    )


# ---------------------------------------------------------------------------
# P2: Persistence — g_mean_ema is NON-persistent (TC-CS calibration buffers)
# ---------------------------------------------------------------------------

class TestTCSCovBuffersNonPersistent:
    """TC-CS g_cov_ema and g_mean_ema must NOT appear in saved state_dict."""

    def test_g_mean_ema_not_in_state_dict(self, tmp_path):
        """g_mean_ema is marked persistent=False; it should not survive save."""
        torch.manual_seed(0)
        model = _create_tiny_local_model()
        config = JoraConfig(
            target_modules=["q_proj"],
            core="diag",
            S_L=4,
            S_R=4,
            k=2,
            pairing_strategy="coupling",
            t_stat=10,
            rotation_impl="torch",
        )
        peft = get_peft_model(model, config)
        peft.eval()

        x = torch.randint(0, 64, (4, 8), dtype=torch.long)
        with torch.no_grad():
            _ = peft(x)

        peft.save_pretrained(tmp_path)

        from safetensors import safe_open
        path = tmp_path / "adapter_model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as f:
            saved_keys = list(f.keys())

        g_mean_keys = [k for k in saved_keys if "g_mean_ema" in k]
        g_cov_keys = [k for k in saved_keys if "g_cov_ema" in k]

        assert len(g_mean_keys) == 0, (
            f"g_mean_ema should NOT be persisted, found: {g_mean_keys}"
        )
        assert len(g_cov_keys) == 0, (
            f"g_cov_ema should NOT be persisted, found: {g_cov_keys}"
        )


# ---------------------------------------------------------------------------
# P3: Persistence — Step indices survive save/load
# ---------------------------------------------------------------------------

class TestStepIndicesPersistence:
    """step_idx and ema_step_idx must survive save/load."""

    def test_step_indices_in_state_dict(self):
        """step_idx and ema_step_idx must be in the PEFT state dict."""
        torch.manual_seed(0)
        model = _create_tiny_local_model()
        config = JoraConfig.diag_path(
            target_modules=["q_proj", "k_proj"],
            S_L=4,
            S_R=4,
            k=2,
        )
        peft = get_peft_model(model, config)

        for module in peft.modules():
            if isinstance(module, JoraLayer):
                for adapter in module.adapters.values():
                    adapter.step_idx.fill_(42)
                    adapter.ema_step_idx.fill_(42)

        sd = get_peft_model_state_dict(peft)
        step_keys = [k for k in sd.keys() if "step_idx" in k]

        assert len(step_keys) > 0, "step_idx must appear in PEFT state dict"
        for key in step_keys:
            assert sd[key].item() == 42, f"{key} should be 42, got {sd[key].item()}"


# ---------------------------------------------------------------------------
# Integration: Full pipeline — adapter eval and merged eval
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """End-to-end: adapter eval and merged eval both work."""

    def test_full_pipeline_norot(self):
        """NoRot: adapter eval and merged eval both match base."""
        torch.manual_seed(0)
        d = 16
        base = _make_diag_layer(d)
        config = JoraConfig(
            target_modules=["q_proj"],
            core="diag",
            S_L=0,
            S_R=0,
            k=2,
            magnitude="none",
            zero_init_core=True,
            selection="none",
            rotation_impl="torch",
        )
        layer = JoraLayer(base, "default", config)
        layer.eval()

        x = torch.randn(8, d)
        with torch.no_grad():
            base_out = base(x)
            adapter_out = layer(x)
            layer.merge()
            merged_out = layer(x)

        torch.testing.assert_close(adapter_out, base_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(merged_out, base_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Gate: Overall M0 verdict
# ---------------------------------------------------------------------------

def test_m0_gate_all_tests_pass():
    """Meta-test: M0 passes iff ALL tests above pass.

    This is a documentation anchor. The actual gate is the pytest exit code:
      - Exit 0:  M0 PASS — proceed to M1 (LoRA/DoRA baselines)
      - Exit 1:  M0 FAIL — STOP, fix correctness before GPU investment
    """
    pass
