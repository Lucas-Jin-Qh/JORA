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

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import OPTConfig, AutoModelForCausalLM, Trainer, TrainingArguments

from peft import JoraConfig, PeftModel, get_peft_model, get_peft_config_from_string
from peft.tuners.jora.callbacks import JoraTrainerCallback
from peft.tuners.jora.magnitude import compute_oer_scale_softmax
from peft.tuners.jora.rotation import apply_rotations_torch
from peft.tuners.jora.selection import select_top_k_pairs_gpu
from peft.utils import infer_device


def create_tiny_local_model():
    """Create a tiny local causal language model without downloading from HuggingFace Hub.

    Returns:
        A tiny OPTForCausalLM model instantiated from local config.
    """
    # Create a minimal config for tiny model using OPTConfig
    hf_config = OPTConfig(
        vocab_size=64,  # Very small vocab
        hidden_size=32,  # Very small hidden size
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        ffn_dim=64,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    # Create model from config (no network access needed)
    model = AutoModelForCausalLM.from_config(hf_config)

    return model


class TestJora:
    device = infer_device()

    def test_jora_config_creation(self):
        """Test JORA config creation."""
        config = JoraConfig(
            target_modules=["q_proj", "k_proj"],
            S_L=8,
            S_R=8,
            k=4,
            rotation_param="cayley",
            rotation_impl="torch",
            selection="topk_ema",
            magnitude="ecd_tanh",
        )
        assert config.peft_type.value == "JORA"
        assert config.S_L == 8
        assert config.S_R == 8
        assert config.k == 4

    def test_jora_config_from_string(self):
        """Test JORA config creation from string."""
        config = get_peft_config_from_string(
            "jora",
            target_modules=["q_proj", "k_proj"],
            S_L=16,
            S_R=16,
            k=8,
        )
        assert config.peft_type.value == "JORA"
        assert config.S_L == 16
        assert config.S_R == 16
        assert config.k == 8

    def test_jora_model_application(self):
        """Test JORA model application."""
        torch.manual_seed(0)

        # Use local tiny model instead of downloading from HuggingFace Hub
        model = create_tiny_local_model().to(self.device)
        model.eval()

        config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=4,
            S_R=4,
            k=2,
            rotation_impl="torch",
        )

        peft_model = get_peft_model(model, config)
        peft_model.eval()

        # Check that PEFT model is created
        assert isinstance(peft_model, PeftModel)

        # Check trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        assert trainable_params > 0, "Should have trainable parameters"

        # Test forward pass
        inputs = torch.arange(10).view(-1, 1).to(self.device)
        outputs = peft_model(inputs)
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == inputs.shape[0]

    def test_jora_config_serialization(self, tmp_path):
        """Test JORA config serialization and deserialization."""
        config = JoraConfig(
            target_modules=["q_proj", "k_proj"],
            S_L=8,
            S_R=8,
            k=4,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert "peft_type" in config_dict
        assert config_dict["peft_type"] == "JORA"

        # Test from_dict
        from peft import get_peft_config
        loaded_config = get_peft_config(config_dict)
        assert loaded_config.peft_type.value == "JORA"
        assert loaded_config.S_L == 8
        assert loaded_config.S_R == 8
        assert loaded_config.k == 4

    def test_jora_save_load(self, tmp_path):
        """Test JORA model save and load."""
        torch.manual_seed(0)

        # Use local tiny model instead of downloading from HuggingFace Hub
        model = create_tiny_local_model().to(self.device)

        config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=4,
            S_R=4,
            k=2,
            rotation_impl="torch",
        )

        peft_model = get_peft_model(model, config)

        # Save model
        peft_model.save_pretrained(tmp_path)

        # Load model - create a fresh base model with SAME random seed
        torch.manual_seed(0)  # Reset seed to get same base weights
        fresh_model = create_tiny_local_model().to(self.device)
        loaded_model = PeftModel.from_pretrained(fresh_model, tmp_path)
        peft_model.eval()
        loaded_model.eval()

        # Check that loaded model works
        inputs = torch.arange(5).view(-1, 1).to(self.device)
        with torch.no_grad():
            original_output = peft_model(inputs)
            loaded_output = loaded_model(inputs)

        assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-5)

    def test_jora_different_rotation_params(self):
        """Test JORA with different rotation parameters."""
        config_cayley = JoraConfig(rotation_param="cayley")
        config_angle = JoraConfig(rotation_param="angle")

        assert config_cayley.rotation_param == "cayley"
        assert config_angle.rotation_param == "angle"

    def test_jora_different_selection_types(self):
        """Test JORA with different selection types."""
        config_topk = JoraConfig(selection="topk_ema")
        config_random = JoraConfig(selection="random")
        config_none = JoraConfig(selection="none")

        assert config_topk.selection == "topk_ema"
        assert config_random.selection == "random"
        assert config_none.selection == "none"

    def test_base_row_norms_nn_linear(self):
        """Test that base_row_norms is correctly computed for nn.Linear layers.

        For nn.Linear with weight shape [out_features, in_features],
        the row norms (dim=1) should match the expected output dimension norms.
        """
        torch.manual_seed(42)

        # Create a simple nn.Linear layer
        in_features, out_features = 16, 32
        linear = nn.Linear(in_features, out_features)

        # Manually compute expected row norms (dim=1 for nn.Linear [out, in])
        # Each row corresponds to one output feature
        expected_row_norms = torch.norm(linear.weight, p=2, dim=1)

        # Apply JORA with magnitude enabled
        config = JoraConfig(
            target_modules=["linear"],
            magnitude="oer_softmax",
            S_L=4,
            S_R=4,
            k=2,
            rotation_impl="torch",
        )

        # Create a minimal model with this linear layer
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear

        model = SimpleModel()
        peft_model = get_peft_model(model, config, adapter_name="test")

        # Find the JoraLayer
        jora_layer = None
        for module in peft_model.modules():
            if "JoraLayer" in type(module).__name__:
                jora_layer = module
                break

        assert jora_layer is not None, "JoraLayer not found"

        # Get adapter state to access base_row_norms
        adapter_state = jora_layer.adapters["test"]

        # Check that base_row_norms matches expected row norms
        cached_norms = adapter_state.base_row_norms
        assert cached_norms is not None, "base_row_norms should be cached for magnitude != 'none'"
        torch.testing.assert_close(cached_norms, expected_row_norms, rtol=1e-5, atol=1e-5)

    def test_merge_unmerge_magnitude_none(self):
        """Test merge/unmerge with magnitude='none'."""
        torch.manual_seed(42)

        model = create_tiny_local_model().to(self.device)
        config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=2,
            S_R=2,
            k=1,
            magnitude="none",  # No magnitude scaling
            rotation_impl="torch",
        )

        peft_model = get_peft_model(model, config, adapter_name="test")
        peft_model.eval()

        # Get original output
        inputs = torch.arange(8).view(-1, 1).to(self.device)
        with torch.no_grad():
            original_output = peft_model(inputs).logits.clone()

        # Merge adapter
        peft_model.merge_adapter(["test"])

        # Get merged output
        with torch.no_grad():
            merged_output = peft_model(inputs).logits.clone()

        # Unmerge adapter
        peft_model.unmerge_adapter()

        # Get unmerged output
        with torch.no_grad():
            unmerged_output = peft_model(inputs).logits.clone()

        # Check that unmerged output matches original
        torch.testing.assert_close(unmerged_output, original_output, rtol=1e-5, atol=1e-5)

    def test_select_top_k_pairs_returns_disjoint_greedy_pairs(self):
        """Selection must return pairwise-disjoint pairs in descending greedy order."""
        energy = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])

        pairs = select_top_k_pairs_gpu(energy, k=4, max_features=energy.numel())

        expected = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        normalized = torch.sort(pairs, dim=1).values

        torch.testing.assert_close(normalized.cpu(), expected)

    def test_apply_rotations_rejects_overlapping_pairs(self):
        """Rotation path should fail fast if selection emits overlapping pairs."""
        x = torch.randn(2, 8)
        pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        thetas = torch.zeros(2)

        with pytest.raises(ValueError, match="overlapping indices"):
            apply_rotations_torch(x, pairs, thetas)

    def test_merge_unmerge_magnitude_oer_softmax(self):
        """Test merge/unmerge with magnitude='oer_softmax'."""
        torch.manual_seed(42)

        model = create_tiny_local_model().to(self.device)
        config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=2,
            S_R=2,
            k=1,
            magnitude="oer_softmax",
            oer_temperature=1.0,
            rotation_impl="torch",
        )

        peft_model = get_peft_model(model, config, adapter_name="test")
        peft_model.eval()

        # Get original output
        inputs = torch.arange(8).view(-1, 1).to(self.device)
        with torch.no_grad():
            original_output = peft_model(inputs).logits.clone()

        # Merge adapter
        peft_model.merge_adapter(["test"])

        # Get merged output
        with torch.no_grad():
            merged_output = peft_model(inputs).logits.clone()

        # Unmerge adapter
        peft_model.unmerge_adapter()

        # Get unmerged output
        with torch.no_grad():
            unmerged_output = peft_model(inputs).logits.clone()

        # Check that unmerged output matches original within tolerance
        max_diff = (unmerged_output - original_output).abs().max()
        assert max_diff < 1e-4, f"Unmerge error too large: {max_diff}"

    def test_oer_softmax_initialization_is_identity(self):
        """OER should preserve the base output at init when delta path is zero."""
        torch.manual_seed(42)

        linear = nn.Linear(16, 32).to(self.device)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear

        model = SimpleModel().to(self.device)
        config = JoraConfig(
            target_modules=["linear"],
            magnitude="oer_softmax",
            oer_temperature=1.0,
            selection="none",
            S_L=0,
            S_R=0,
            k=0,
            zero_init_core=True,
            rotation_impl="torch",
        )

        peft_model = get_peft_model(model, config, adapter_name="test")

        jora_layer = None
        for module in peft_model.modules():
            if "JoraLayer" in type(module).__name__:
                jora_layer = module
                break

        assert jora_layer is not None, "JoraLayer not found"

        adapter_state = jora_layer.adapters["test"]
        scale = compute_oer_scale_softmax(
            base_row_norms=adapter_state.base_row_norms_fp32,
            total_energy=adapter_state.total_energy,
            oer_logits=adapter_state.ecd_log_mag,
            temperature=adapter_state.cfg.oer_temperature,
            eps=adapter_state.cfg.eps,
        )
        torch.testing.assert_close(scale, torch.ones_like(scale), rtol=1e-5, atol=1e-5)

        x = torch.randn(4, 16, device=self.device)
        with torch.no_grad():
            base_out = jora_layer.base_layer(x)
            jora_out = jora_layer(x)

        torch.testing.assert_close(jora_out, base_out, rtol=1e-5, atol=1e-5)

    def test_jora_trainer_callback_smoke(self, tmp_path):
        """Run a tiny offline Trainer loop to validate callback-driven updates."""
        torch.manual_seed(1234)

        model = create_tiny_local_model()
        config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=4,
            S_R=4,
            k=2,
            selection="topk_ema",
            magnitude="oer_softmax",
            rotation_impl="torch",
            warmup_steps=2,
        )
        peft_model = get_peft_model(model, config)
        peft_model.train()

        seqs = [
            [1, 5, 6, 7, 2, 0],
            [1, 8, 9, 10, 2, 0],
            [1, 11, 12, 13, 2, 0],
            [1, 14, 15, 16, 2, 0],
            [1, 17, 18, 19, 2, 0],
            [1, 20, 21, 22, 2, 0],
            [1, 23, 24, 25, 2, 0],
            [1, 26, 27, 28, 2, 0],
        ]
        attention_mask = [[1, 1, 1, 1, 1, 0] for _ in seqs]
        labels = [row[:] for row in seqs]
        train_dataset = Dataset.from_dict(
            {"input_ids": seqs, "attention_mask": attention_mask, "labels": labels}
        )

        def collator(features):
            return {
                key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
                for key in features[0]
            }

        params_before = {
            name: param.detach().cpu().clone()
            for name, param in peft_model.named_parameters()
            if param.requires_grad
        }

        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=4,
            learning_rate=5e-3,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            use_cpu=str(self.device) == "cpu",
        )
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
            callbacks=[JoraTrainerCallback(peft_model, verbose=False)],
        )
        result = trainer.train()

        changed = []
        for name, param in peft_model.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.allclose(params_before[name], param.detach().cpu()):
                changed.append(name)

        assert changed, "Trainer loop should update at least one JORA parameter"
        assert result.training_loss > 0
        assert peft_model.base_model._jora_global_step == 4
        assert peft_model.base_model._jora_total_steps == 4
