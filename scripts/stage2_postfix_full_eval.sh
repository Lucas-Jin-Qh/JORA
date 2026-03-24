#!/usr/bin/env bash
# Post-fix Full MMLU Evaluation
# GPU0: JORA-none (full eval)
# GPU1: JORA-oer_softmax (full eval)
# GPU2: LoRA-r4 (full eval)
#
# Results will be saved to /tmp/jora-postfix-stage2/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src}"

WORKDIR="${WORKDIR:-/tmp/jora-postfix-stage2}"
MODEL_ID="${MODEL_ID:-facebook/opt-350m}"

# Existing post-fix model paths
JORA_NONE_PATH="${JORA_NONE_PATH:-/tmp/jora-eval-full/out_jora_none}"
JORA_OER_PATH="${JORA_OER_PATH:-/tmp/jora-eval-full/out_jora_oer}"
LORA_R4_PATH="${LORA_R4_PATH:-/tmp/jora-baseline/out_lora_r4}"

mkdir -p "$WORKDIR"

echo "=== Post-fix Full MMLU Evaluation ==="
echo "Model: $MODEL_ID"
echo "Workdir: $WORKDIR"
echo ""

# Get base model from adapter config
get_base_model() {
    local adapter_config="$1/adapter_config.json"
    if [ -f "$adapter_config" ]; then
        grep -o '"base_model_name_or_path"[[:space:]]*:[[:space:]]*"[^"]*"' "$adapter_config" | sed 's/.*"\([^"]*\)"$/\1/'
    else
        echo "$MODEL_ID"
    fi
}

# Get base model from first model path (they should all be the same)
ACTUAL_MODEL_ID=$(get_base_model "$JORA_NONE_PATH")
echo "Detected base model: $ACTUAL_MODEL_ID"

echo "Model paths:"
echo "  JORA-none: $JORA_NONE_PATH"
echo "  JORA-oer: $JORA_OER_PATH"
echo "  LoRA-r4: $LORA_R4_PATH"
echo ""

# Check if models exist
for path in "$JORA_NONE_PATH" "$JORA_OER_PATH" "$LORA_R4_PATH"; do
    if [ ! -d "$path" ]; then
        echo "ERROR: Model not found: $path"
        exit 1
    fi
done

# Run full MMLU evaluation (all samples)
run_eval_gpu() {
    local gpu_id="$1"
    local model_path="$2"
    local output_name="$3"
    local log_file="$WORKDIR/${output_name}_full_eval.log"
    
    echo ">>> GPU$gpu_id: Evaluating $output_name (full MMLU)"
    echo "    Model: $model_path"
    echo "    Log: $log_file"
    
    # Record env vars in log
    {
        echo "=== Environment ==="
        echo "MODEL_PATH: $model_path"
        echo "OUTPUT_NAME: $output_name"
        echo "GPU: $gpu_id"
        echo "MODEL_ID: $MODEL_ID"
        echo ""
    } > "$log_file"
    
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" - <<EOF 2>&1 | tee -a "$log_file"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

model_path = "$model_path"
output_name = "$output_name"
MODEL_ID = "$ACTUAL_MODEL_ID"

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

# Load full MMLU test set
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test")
num_samples = len(ds)
print(f"Full MMLU test samples: {num_samples}")

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
print(f"Full MMLU Accuracy: {accuracy:.4f} ({correct}/{total})")

# Save result
with open("$WORKDIR/results_full.txt", "a") as f:
    f.write(f"{output_name}: {accuracy:.4f} ({correct}/{total})\n")

print(f"Result saved to $WORKDIR/results_full.txt")
EOF
    
    echo "<<< GPU$gpu_id: Done"
}

# Initialize results file
echo "=== Full MMLU Results ===" > "$WORKDIR/results_full.txt"
echo "Date: $(date)" >> "$WORKDIR/results_full.txt"
echo "Model: $MODEL_ID" >> "$WORKDIR/results_full.txt"
echo "" >> "$WORKDIR/results_full.txt"

echo "Starting parallel evaluations..."
echo ""

# Run evaluations in parallel using background jobs
run_eval_gpu 0 "$JORA_NONE_PATH" "JORA-none" &
PID0=$!

run_eval_gpu 1 "$JORA_OER_PATH" "JORA-oer_softmax" &
PID1=$!

run_eval_gpu 2 "$LORA_R4_PATH" "LoRA-r4" &
PID2=$!

echo "Launched 3 evaluation jobs:"
echo "  GPU0 (PID=$PID0): JORA-none"
echo "  GPU1 (PID=$PID1): JORA-oer_softmax"
echo "  GPU2 (PID=$PID2): LoRA-r4"
echo ""
echo "Waiting for completion..."

# Wait for all to complete
wait $PID0
wait $PID1
wait $PID2

echo ""
echo "=========================================="
echo "All Evaluations Complete"
echo "=========================================="
echo ""
echo "Results:"
cat "$WORKDIR/results_full.txt"
echo ""
echo "Log files:"
ls -la "$WORKDIR"/*.log
