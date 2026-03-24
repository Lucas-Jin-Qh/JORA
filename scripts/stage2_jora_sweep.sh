#!/usr/bin/env bash
# Stage 2: JORA Sweep on opt-350m
# Tests 7 JORA configurations + LoRA-r4 baseline
# Phase 1: Coarse eval (1000 samples) -> Phase 2: Fine eval top-2 (2000 samples)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src"

# Model and training config
MODEL_ID="${MODEL_ID:-facebook/opt-350m}"
DATASET_NAME="${DATASET_NAME:-yahma/alpaca-cleaned}"
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WORKDIR="${WORKDIR:-/tmp/jora-sweep}"
NUM_EVAL_SAMPLES_COARSE="${NUM_EVAL_SAMPLES_COARSE:-1000}"
NUM_EVAL_SAMPLES_FINE="${NUM_EVAL_SAMPLES_FINE:-2000}"

# JORA params to sweep
JORA_TARGETS="${JORA_TARGETS:-q_proj,k_proj,v_proj,out_proj}"
JORA_SELECTION="${JORA_SELECTION:-topk_ema}"
JORA_WARMUP="${JORA_WARMUP:-50}"

# LoRA config for baseline
LORA_R=4

echo "=== Stage 2: JORA Sweep ==="
echo "Model: $MODEL_ID"
echo "Steps: $MAX_STEPS, LR: $LEARNING_RATE"
echo "Coarse eval: $NUM_EVAL_SAMPLES_COARSE samples"
echo "Fine eval: $NUM_EVAL_SAMPLES_FINE samples"
echo ""

mkdir -p "$WORKDIR"
RESULTS_COARSE="$WORKDIR/results_coarse.txt"
RESULTS_FINE="$WORKDIR/results_fine.txt"
touch "$RESULTS_COARSE" "$RESULTS_FINE"

has_result() {
    local result_file="$1"
    local output_name="$2"
    [ -f "$result_file" ] && grep -q "^${output_name}:" "$result_file"
}

# Training function for JORA
train_jora() {
    local output_dir="$1"
    local s_l="$2"
    local s_r="$3"
    local k="$4"
    local magnitude="$5"
    local lr="$6"
    
    if [ -d "$output_dir" ]; then
        echo "Using existing: $output_dir"
        return 0
    fi
    
    echo "=== Training JORA S_L=$s_l S_R=$s_r k=$k mag=$magnitude lr=$lr ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" examples/sft/train.py \
        --seed 42 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits train \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$MAX_STEPS" \
        --logging_steps 100 \
        --save_steps 500 \
        --eval_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$lr" \
        --max_length 512 \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_jora True \
        --lora_target_modules "$JORA_TARGETS" \
        --jora_s_l "$s_l" \
        --jora_s_r "$s_r" \
        --jora_k "$k" \
        --jora_rotation_impl auto \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude "$magnitude" \
        --jora_warmup_steps "$JORA_WARMUP" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03
    
    echo "Training complete: $output_dir"
}

# Training function for LoRA
train_lora() {
    local output_dir="$1"
    local r="$2"
    local lora_alpha=$((r * 2))
    
    if [ -d "$output_dir" ]; then
        echo "Using existing: $output_dir"
        return 0
    fi
    
    echo "=== Training LoRA r=$r, alpha=$lora_alpha, dropout=0.0 ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" examples/sft/train.py \
        --seed 42 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits train \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$MAX_STEPS" \
        --logging_steps 100 \
        --save_steps 500 \
        --eval_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$LEARNING_RATE" \
        --max_length 512 \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_lora True \
        --lora_target_modules "$JORA_TARGETS" \
        --lora_r "$r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout 0.0 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03
    
    echo "Training complete: $output_dir"
}

# Evaluation function
eval_model() {
    local model_path="$1"
    local output_name="$2"
    local num_samples="$3"
    local result_file="$4"
    
    echo "=== Evaluating $output_name on $num_samples samples ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<PYEOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

MODEL_ID = "$MODEL_ID"
model_path = "$model_path"
num_samples = $num_samples
output_name = "$output_name"
result_file = "$result_file"

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
    
    if (i + 1) % 200 == 0:
        print(f"  Progress: {i+1}/{total}, Accuracy: {correct/total:.4f}")

accuracy = correct / total
print(f"\n=== {output_name} ===")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

with open(result_file, "a") as f:
    f.write(f"{output_name}: {accuracy:.4f} ({correct}/{total})\n")
PYEOF
}

# Configurations to test (name:s_l:s_r:k:magnitude:lr:note)
declare -a JORA_CONFIGS=(
    "sweep_1:8:8:8:oer_softmax:1e-4:基准"
    "sweep_2:4:4:8:oer_softmax:1e-4:低容量"
    "sweep_3:16:16:8:oer_softmax:1e-4:高容量"
    "sweep_4:8:8:4:oer_softmax:1e-4:低rank"
    "sweep_5:8:8:16:oer_softmax:1e-4:高rank"
    "sweep_6:8:8:8:none:1e-4:magnitude对照"
    "sweep_7:8:8:8:oer_softmax:5e-5:低lr对照"
)

echo "=========================================="
echo "Phase 1: Coarse Evaluation (1000 samples)"
echo "=========================================="
echo ""

# Train and eval LoRA baseline first
echo ">>> Training LoRA-r$LORA_R baseline"
train_lora "$WORKDIR/out_lora_r$LORA_R" "$LORA_R"
if has_result "$RESULTS_COARSE" "lora_r$LORA_R"; then
    echo "Skipping coarse eval for lora_r$LORA_R (already recorded)"
else
    eval_model "$WORKDIR/out_lora_r$LORA_R" "lora_r$LORA_R" "$NUM_EVAL_SAMPLES_COARSE" "$RESULTS_COARSE"
fi

# Train and eval JORA configs
for config in "${JORA_CONFIGS[@]}"; do
    IFS=':' read -r name s_l s_r k mag lr note <<< "$config"
    output_dir="$WORKDIR/$name"
    
    echo ""
    echo ">>> Processing $name ($note)"
    train_jora "$output_dir" "$s_l" "$s_r" "$k" "$mag" "$lr"
    if has_result "$RESULTS_COARSE" "$name"; then
        echo "Skipping coarse eval for $name (already recorded)"
    else
        eval_model "$output_dir" "$name" "$NUM_EVAL_SAMPLES_COARSE" "$RESULTS_COARSE"
    fi
    
    result=$(grep "^${name}:" "$RESULTS_COARSE" | tail -1)
    echo "Result: $result"
done

echo ""
echo "=========================================="
echo "Coarse Results Summary"
echo "=========================================="
cat "$RESULTS_COARSE"
echo ""

# Find top-2 JORA configs (excluding lora)
echo "Finding top-2 JORA configs..."
TOP2=$(grep "^sweep_" "$RESULTS_COARSE" | sort -t':' -k2 -rn | head -2 | awk -F':' '{print $1}')
TOP2_ARRAY=($TOP2)

echo "Top-2 JORA configs: ${TOP2_ARRAY[0]}, ${TOP2_ARRAY[1]}"

echo ""
echo "=========================================="
echo "Phase 2: Fine Evaluation (2000 samples)"
echo "=========================================="
echo ""

# Re-evaluate top-2 JORA with 2000 samples
for name in "${TOP2_ARRAY[@]}"; do
    output_dir="$WORKDIR/$name"
    echo ">>> Fine eval $name"
    if has_result "$RESULTS_FINE" "${name}_fine"; then
        echo "Skipping fine eval for ${name}_fine (already recorded)"
    else
        eval_model "$output_dir" "${name}_fine" "$NUM_EVAL_SAMPLES_FINE" "$RESULTS_FINE"
    fi
    
    result=$(grep "^${name}_fine:" "$RESULTS_FINE" | tail -1)
    echo "Fine result: $result"
done

# Also re-evaluate LoRA with 2000 samples for fair comparison
echo ">>> Fine eval lora_r$LORA_R"
if has_result "$RESULTS_FINE" "lora_r${LORA_R}_fine"; then
    echo "Skipping fine eval for lora_r${LORA_R}_fine (already recorded)"
else
    eval_model "$WORKDIR/out_lora_r$LORA_R" "lora_r${LORA_R}_fine" "$NUM_EVAL_SAMPLES_FINE" "$RESULTS_FINE"
fi

echo ""
echo "=========================================="
echo "Final Results Summary"
echo "=========================================="
echo ""
echo "Coarse (1000 samples):"
cat "$RESULTS_COARSE"
echo ""
echo "Fine (2000 samples):"
cat "$RESULTS_FINE"
echo ""

# Final ranking from fine eval
echo "Final Ranking (2000 samples):"
grep "_fine:" "$RESULTS_FINE" | sort -t':' -k2 -rn

echo ""
echo "=========================================="
echo "Sweep Complete"
echo "=========================================="
echo "Top-2 JORA configs to proceed to Stage 3:"
grep "^sweep_" "$WORKDIR/results_fine.txt" | sort -t':' -k2 -rn | head -2 | awk -F':' '{print "  - " $1}'
