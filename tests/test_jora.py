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

import torch
from transformers import AutoModelForCausalLM

from peft import JoraConfig, PeftModel, get_peft_model, get_peft_config_from_string
from peft.utils import infer_device


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
            selection_type="topk_ema",
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

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model.eval()

        config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=4,
            S_R=4,
            k=2,
            init_weights=False,  # Use deterministic weights for testing
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

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = JoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            S_L=4,
            S_R=4,
            k=2,
            init_weights=False,
        )

        peft_model = get_peft_model(model, config)

        # Save model
        peft_model.save_pretrained(tmp_path)

        # Load model
        loaded_model = PeftModel.from_pretrained(model, tmp_path)

        # Check that loaded model works
        inputs = torch.arange(5).view(-1, 1).to(self.device)
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
        config_topk = JoraConfig(selection_type="topk_ema")
        config_random = JoraConfig(selection_type="random")
        config_none = JoraConfig(selection_type="none")

        assert config_topk.selection_type == "topk_ema"
        assert config_random.selection_type == "random"
        assert config_none.selection_type == "none"
