#!/usr/bin/env bash
# Stage 6: Evaluation matrix - JORA oer_softmax vs none vs LoRA r=1
# Tests on MMLU (multi-task language understanding)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"

export PYTHONPATH="${REPO_ROOT}/src"

MODEL_ID="${MODEL_ID:-facebook/opt-125m}"
DATASET_NAME="${DATASET_NAME:-yahma/alpaca-cleaned}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_STEPS="${MAX_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
WORKDIR="${WORKDIR:-/tmp/jora-eval}"
EVAL_DATASET="${EVAL_DATASET:-mmlu}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-100}"

JORA_TARGETS="${JORA_TARGETS:-q_proj,k_proj,v_proj,out_proj}"
JORA_S_L="${JORA_S_L:-16}"
JORA_S_R="${JORA_S_R:-16}"
JORA_K="${JORA_K:-8}"
JORA_SELECTION="${JORA_SELECTION:-topk_ema}"
JORA_MAGNITUDE="${JORA_MAGNITUDE:-oer_softmax}"
JORA_WARMUP="${JORA_WARMUP:-10}"

echo "=== Stage 6: Evaluation Matrix ==="
echo "Training: $MAX_STEPS steps on $DATASET_NAME"
echo "Evaluation: $EVAL_DATASET ($NUM_EVAL_SAMPLES samples)"
echo ""

mkdir -p "$WORKDIR"

ensure_absent() {
    if [ -d "$1" ]; then
        echo "ERROR: $1 already exists, delete first"
        exit 1
    fi
}

print_summary() {
    echo "============================================"
    echo "Model: $MODEL_ID"
    echo "Dataset: $DATASET_NAME (${DATASET_SPLIT})"
    echo "Steps: $MAX_STEPS, LR: $LEARNING_RATE"
    echo "Eval: $EVAL_DATASET"
    echo "============================================"
}

# Train JORA with oer_softmax
train_jora_oer() {
    local output_dir="$1"
    ensure_absent "$output_dir"

    echo "=== Training JORA oer_softmax ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" examples/sft/train.py \
        --seed 1234 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$MAX_STEPS" \
        --logging_steps 50 \
        --eval_strategy no \
        --save_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_jora True \
        --lora_target_modules "$JORA_TARGETS" \
        --jora_s_l "$JORA_S_L" \
        --jora_s_r "$JORA_S_R" \
        --jora_k "$JORA_K" \
        --jora_rotation_impl auto \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude oer_softmax \
        --jora_warmup_steps "$JORA_WARMUP"
}

# Train JORA with magnitude=none
train_jora_none() {
    local output_dir="$1"
    ensure_absent "$output_dir"

    echo "=== Training JORA magnitude=none ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" examples/sft/train.py \
        --seed 1234 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$MAX_STEPS" \
        --logging_steps 50 \
        --eval_strategy no \
        --save_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_jora True \
        --lora_target_modules "$JORA_TARGETS" \
        --jora_s_l "$JORA_S_L" \
        --jora_s_r "$JORA_S_R" \
        --jora_k "$JORA_K" \
        --jora_rotation_impl auto \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude none \
        --jora_warmup_steps "$JORA_WARMUP"
}

# Train LoRA baseline
train_lora() {
    local output_dir="$1"
    local r="$2"
    ensure_absent "$output_dir"

    local lora_alpha=$((r * 2))
    echo "=== Training LoRA r=$r, alpha=$lora_alpha ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" examples/sft/train.py \
        --seed 1234 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$MAX_STEPS" \
        --logging_steps 50 \
        --eval_strategy no \
        --save_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_lora True \
        --lora_target_modules "$JORA_TARGETS" \
        --lora_r "$r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout 0.0
}

# Evaluate on MMLU
run_eval() {
    local model_path="$1"
    local output_name="$2"

    echo "=== Evaluating $output_name on MMLU ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

print(f"Loading base model: $MODEL_ID")
base = AutoModelForCausalLM.from_pretrained(
    "$MODEL_ID",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading adapter: $model_path")
model = PeftModel.from_pretrained(base, "$model_path")
model.eval()

# Load MMLU
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test")
num_samples = $NUM_EVAL_SAMPLES
ds = ds.select(range(num_samples))

def format_prompt(row):
    # Simple format: question + choices
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
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{total}, Accuracy: {correct/total:.4f}")

accuracy = correct / total
print(f"\n=== $output_name ===")
print(f"MMLU Accuracy ({num_samples} samples): {accuracy:.4f}")
print(f"Correct: {correct}/{total}")

# Save result
with open("$WORKDIR/eval_results.txt", "a") as f:
    f.write(f"$output_name: {accuracy:.4f} ({correct}/{total})\n")
EOF
}

print_help() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  jora_oer      - Train JORA with oer_softmax and evaluate"
    echo "  jora_none     - Train JORA with magnitude=none and evaluate"
    echo "  lora_r1       - Train LoRA r=1 and evaluate"
    echo "  all           - Run all three and compare"
    echo "  eval <path>   - Evaluate existing checkpoint"
    echo ""
    echo "Environment variables:"
    echo "  MAX_STEPS      - Training steps (default: 500)"
    echo "  NUM_EVAL_SAMPLES - MMLU samples (default: 100)"
    echo "  WORKDIR        - Output directory"
}

case "${1:-}" in
    jora_oer)
        print_summary
        train_jora_oer "$WORKDIR/out_jora_oer"
        run_eval "$WORKDIR/out_jora_oer" "JORA-oer_softmax"
        ;;
    jora_none)
        print_summary
        train_jora_none "$WORKDIR/out_jora_none"
        run_eval "$WORKDIR/out_jora_none" "JORA-none"
        ;;
    lora_r1)
        print_summary
        train_lora "$WORKDIR/out_lora_r1" 1
        run_eval "$WORKDIR/out_lora_r1" "LoRA-r1"
        ;;
    all)
        print_summary
        train_jora_oer "$WORKDIR/out_jora_oer"
        run_eval "$WORKDIR/out_jora_oer" "JORA-oer_softmax"
        
        train_jora_none "$WORKDIR/out_jora_none"
        run_eval "$WORKDIR/out_jora_none" "JORA-none"
        
        train_lora "$WORKDIR/out_lora_r1" 1
        run_eval "$WORKDIR/out_lora_r1" "LoRA-r1"
        
        echo ""
        echo "=== FINAL RESULTS ==="
        cat "$WORKDIR/eval_results.txt"
        ;;
    eval)
        if [ -z "${2:-}" ]; then
            echo "ERROR: provide model path"
            exit 1
        fi
        run_eval "$2" "$(basename $2)"
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo "Unknown command: ${1:-}"
        print_help
        exit 1
        ;;
esac
