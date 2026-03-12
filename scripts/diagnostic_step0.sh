#!/usr/bin/env bash
# Step-0 diagnostic: compare base model vs fresh JORA loss/logits drift

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"

export PYTHONPATH="${REPO_ROOT}/src"

MODEL_ID="${MODEL_ID:-facebook/opt-125m}"

echo "=== Step-0 Diagnostic: Base vs JORA ==="
echo "Model: $MODEL_ID"
echo ""

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import JoraConfig, get_peft_model

print("Loading base model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
base = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    torch_dtype=torch.bfloat16,
).to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokenizer.pad_token = tokenizer.eos_token

# Test prompt
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Base model forward
base.eval()
with torch.no_grad():
    base_out = base(**inputs)
    base_logits = base_out.logits
    base_loss = torch.nn.functional.cross_entropy(
        base_logits.view(-1, base_logits.size(-1)),
        inputs["input_ids"].view(-1),
        ignore_index=tokenizer.pad_token_id,
    )

print(f"Base model loss: {base_loss.item():.4f}")
print(f"Base logits shape: {tuple(base_logits.shape)}")
print(f"Base logits mean: {base_logits.mean().item():.6f}, std: {base_logits.std().item():.6f}")

# Fresh JORA model
print("\n--- Loading fresh JORA model ---")
jora_config = JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    S_L=16,
    S_R=16,
    k=8,
    selection="topk_ema",
    magnitude="oer_softmax",
    oer_temperature=1.0,
)
jora_base = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    torch_dtype=torch.bfloat16,
).to(device)
jora_model = get_peft_model(jora_base, jora_config)
jora_model.eval()

# Get JORA parameters info
total_params = sum(p.numel() for p in jora_model.parameters())
trainable_params = sum(p.numel() for p in jora_model.parameters() if p.requires_grad)
print(f"JORA total params: {total_params:,}")
print(f"JORA trainable params: {trainable_params:,}")

with torch.no_grad():
    jora_out = jora_model(**inputs)
    jora_logits = jora_out.logits
    jora_loss = torch.nn.functional.cross_entropy(
        jora_logits.view(-1, jora_logits.size(-1)),
        inputs["input_ids"].view(-1),
        ignore_index=tokenizer.pad_token_id,
    )

print(f"\nJORA (fresh, T=1.0) loss: {jora_loss.item():.4f}")
print(f"JORA logits mean: {jora_logits.mean().item():.6f}, std: {jora_logits.std().item():.6f}")

# Compute drift
logits_diff = (jora_logits - base_logits).abs().mean().item()
print(f"\nLogits drift (mean abs diff): {logits_diff:.6f}")

# Test with T=2.0
print("\n--- Testing with OER temperature=2.0 ---")
jora_config_2 = JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    S_L=16,
    S_R=16,
    k=8,
    selection="topk_ema",
    magnitude="oer_softmax",
    oer_temperature=2.0,
)
jora_base_2 = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    torch_dtype=torch.bfloat16,
).to(device)
jora_model_2 = get_peft_model(jora_base_2, jora_config_2)
jora_model_2.eval()

with torch.no_grad():
    jora_out_2 = jora_model_2(**inputs)
    jora_loss_2 = torch.nn.functional.cross_entropy(
        jora_out_2.logits.view(-1, jora_out_2.logits.size(-1)),
        inputs["input_ids"].view(-1),
        ignore_index=tokenizer.pad_token_id,
    )

print(f"JORA (fresh, T=2.0) loss: {jora_loss_2.item():.4f}")
logits_diff_2 = (jora_out_2.logits - base_logits).abs().mean().item()
print(f"Logits drift (T=2.0): {logits_diff_2:.6f}")

# Test with magnitude=none
print("\n--- Testing with magnitude=none ---")
jora_config_3 = JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    S_L=16,
    S_R=16,
    k=8,
    selection="topk_ema",
    magnitude="none",
    oer_temperature=1.0,
)
jora_base_3 = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    torch_dtype=torch.bfloat16,
).to(device)
jora_model_3 = get_peft_model(jora_base_3, jora_config_3)
jora_model_3.eval()

with torch.no_grad():
    jora_out_3 = jora_model_3(**inputs)
    jora_loss_3 = torch.nn.functional.cross_entropy(
        jora_out_3.logits.view(-1, jora_out_3.logits.size(-1)),
        inputs["input_ids"].view(-1),
        ignore_index=tokenizer.pad_token_id,
    )

print(f"JORA (fresh, magnitude=none) loss: {jora_loss_3.item():.4f}")
logits_diff_3 = (jora_out_3.logits - base_logits).abs().mean().item()
print(f"Logits drift (magnitude=none): {logits_diff_3:.6f}")

print("\n=== Summary ===")
print(f"Base model loss:         {base_loss.item():.4f}")
print(f"JORA T=1.0 loss:         {jora_loss.item():.4f} (drift: {logits_diff:.6f})")
print(f"JORA T=2.0 loss:         {jora_loss_2.item():.4f} (drift: {logits_diff_2:.6f})")
print(f"JORA magnitude=none loss: {jora_loss_3.item():.4f} (drift: {logits_diff_3:.6f})")
PY
