#!/usr/bin/env bash
# MMLU Evaluation for JORA LR Matrix
# Evaluates 3 JORA models on full MMLU in parallel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src"

MODEL_ID="${MODEL_ID:-facebook/opt-350m}"
WORKDIR="${WORKDIR:-/tmp/jora-lr-matrix}"
NUM_SAMPLES=14042

declare -a EVAL_CONFIGS=(
    "0:/tmp/jora-lr-matrix/lr_1x_1x:lr_1x_1x"
    "1:/tmp/jora-lr-matrix/lr_5x_1x:lr_5x_1x"
    "2:/tmp/jora-lr-matrix/lr_10x_2x:lr_10x_2x"
)

RESULTS_FILE="$WORKDIR/eval_results.txt"
echo "=== Full MMLU Results ===" > "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "Model: $MODEL_ID" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Create eval script
EVAL_SCRIPT="$WORKDIR/eval_mmlu.py"

cat > "$EVAL_SCRIPT" <<'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys

MODEL_ID = sys.argv[1]
model_path = sys.argv[2]
output_name = sys.argv[3]
num_samples = 14042
results_file = sys.argv[4]

print(f"Loading base model: {MODEL_ID}")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

if os.path.exists(f"{model_path}/adapter_config.json"):
    print(f"Loading adapter: {model_path}")
    model = PeftModel.from_pretrained(base, model_path)
else:
    print("Using base model (no adapter)")
    model = base
model.eval()

from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test")
ds = ds.select(range(num_samples))
print(f"Test samples: {num_samples}")

def format_prompt(row):
    question = row["question"]
    choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"])])
    return f"{question}\n{choices}\nAnswer:"

choice_texts = [" A", " B", " C", " D"]

def score_choice(prompt, choice_text):
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(model.device)
    full_ids = tokenizer(prompt + choice_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(model.device)
    choice_ids = full_ids[:, prompt_ids.shape[1]:]
    if choice_ids.numel() == 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        log_probs = outputs.logits[:, :-1, :].log_softmax(dim=-1)

    start = prompt_ids.shape[1] - 1
    end = full_ids.shape[1] - 1
    token_log_probs = log_probs[:, start:end, :]
    gathered = token_log_probs.gather(-1, choice_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.sum().item()

correct = 0
total = 0
for i, row in enumerate(ds):
    prompt = format_prompt(row)
    choice_scores = [score_choice(prompt, choice_text) for choice_text in choice_texts]
    pred_idx = max(range(len(choice_scores)), key=lambda idx: choice_scores[idx])
    true_idx = row["answer"]
    if pred_idx == true_idx:
        correct += 1
    total += 1
    
    if (i + 1) % 1000 == 0:
        print(f"  Progress: {i+1}/{total}, Accuracy: {correct/total:.4f}")

accuracy = correct / total
print(f"\n=== {output_name} ===")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

with open(results_file, "a") as f:
    f.write(f"{output_name}: {accuracy:.4f} ({correct}/{total})\n")

print("Done!")
PYEOF

echo "=== MMLU Evaluation (Full $NUM_SAMPLES samples) ==="
echo ""

# Function to evaluate
eval_model() {
    local gpu_id="$1"
    local model_path="$2"
    local output_name="$3"
    local log_file="$4"
    
    echo "[GPU$gpu_id] Evaluating $output_name..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$PYTHON_BIN" "$EVAL_SCRIPT" "$MODEL_ID" "$model_path" "$output_name" "$RESULTS_FILE" > "$log_file" 2>&1 &
    
    echo "[GPU$gpu_id] Evaluation started (PID: $!), log: $log_file"
}

# Launch 3 evaluations in parallel
for config in "${EVAL_CONFIGS[@]}"; do
    IFS=':' read -r gpu_id model_path output_name <<< "$config"
    log_file="$WORKDIR/${output_name}_eval.log"
    
    eval_model "$gpu_id" "$model_path" "$output_name" "$log_file"
done

echo ""
echo "=========================================="
echo "All 3 evaluations launched in parallel!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tail -f $WORKDIR/lr_1x_1x_eval.log"
echo "  tail -f $WORKDIR/lr_5x_1x_eval.log"
echo "  tail -f $WORKDIR/lr_10x_2x_eval.log"
echo ""
echo "Results will be saved to: $RESULTS_FILE"
