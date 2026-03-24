#!/usr/bin/env bash
# Stage 7: Real evaluation with stronger model + longer training
# - opt-350m or opt-1.3b
# - 2000-5000 steps
# - Full MMLU test set
# - Instruction-following / retention metric

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"

export PYTHONPATH="${REPO_ROOT}/src"

# Model configs
MODEL_ID="${MODEL_ID:-facebook/opt-350m}"
DATASET_NAME="${DATASET_NAME:-yahma/alpaca-cleaned}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_STEPS="${MAX_STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
WORKDIR="${WORKDIR:-/tmp/jora-stage7}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-0}"  # 0 = full test set

# JORA params (parameter-matched to LoRA r=4)
JORA_TARGETS="${JORA_TARGETS:-q_proj,k_proj,v_proj,out_proj}"
JORA_S_L="${JORA_S_L:-8}"
JORA_S_R="${JORA_S_R:-8}"
JORA_K="${JORA_K:-8}"
JORA_SELECTION="${JORA_SELECTION:-topk_ema}"
JORA_MAGNITUDE="${JORA_MAGNITUDE:-oer_softmax}"
JORA_WARMUP="${JORA_WARMUP:-50}"
JORA_LR_THETA="${JORA_LR_THETA:-$LEARNING_RATE}"
JORA_LR_CORE="${JORA_LR_CORE:-$LEARNING_RATE}"

# LoRA r=4 for comparison (similar param count as JORA)
LORA_R=4

echo "=== Stage 7: Real Evaluation ==="
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET_NAME"
echo "Steps: $MAX_STEPS, LR: $LEARNING_RATE"
echo "MMLU samples: ${NUM_EVAL_SAMPLES:-full}"
echo "JORA_LR_THETA: $JORA_LR_THETA"
echo "JORA_LR_CORE: $JORA_LR_CORE"
echo ""

mkdir -p "$WORKDIR"
RESULTS_FILE="$WORKDIR/results.txt"

ensure_absent() {
    if [ -d "$1" ]; then
        echo "Removing existing $1"
        rm -rf "$1"
    fi
}

require_adapter_dir() {
    local adapter_dir="$1"
    if [ ! -d "$adapter_dir" ]; then
        echo "ERROR: adapter directory not found: $adapter_dir"
        exit 1
    fi
    if [ ! -f "$adapter_dir/adapter_config.json" ]; then
        echo "ERROR: missing adapter_config.json in $adapter_dir"
        exit 1
    fi
}

print_summary() {
    echo "============================================"
    echo "Model: $MODEL_ID"
    echo "Dataset: $DATASET_NAME (${DATASET_SPLIT})"
    echo "Steps: $MAX_STEPS, LR: $LEARNING_RATE"
    echo "============================================"
}

# Train JORA
train_jora() {
    local output_dir="$1"
    local magnitude="${2:-oer_softmax}"
    ensure_absent "$output_dir"

    echo "=== Training JORA magnitude=$magnitude ==="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" examples/sft/train.py \
        --seed 42 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
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
        --jora_magnitude "$magnitude" \
        --jora_warmup_steps "$JORA_WARMUP" \
        --jora_lr_theta "$JORA_LR_THETA" \
        --jora_lr_core "$JORA_LR_CORE" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03
}

# Train LoRA
train_lora() {
    local output_dir="$1"
    local r="${2:-4}"
    ensure_absent "$output_dir"

    local lora_alpha=$((r * 2))
    echo "=== Training LoRA r=$r, alpha=$lora_alpha ==="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" examples/sft/train.py \
        --seed 42 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
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
        --max_length "$MAX_LENGTH" \
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
}

# Evaluate on MMLU (full test set)
run_mmlu_eval() {
    local model_path="$1"
    local output_name="$2"
    local num_samples="$3"

    echo "=== Evaluating $output_name on MMLU ==="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" - <<EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

# Load MMLU full test set
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test")
print(f"MMLU test samples: {len(ds)}")

if $num_samples > 0:
    ds = ds.select(range($num_samples))
    print(f"Using first $num_samples samples")

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
print(f"\n=== $output_name ===")
print(f"MMLU Accuracy ({total} samples): {accuracy:.4f}")
print(f"Correct: {correct}/{total}")

# Save result
with open("$RESULTS_FILE", "a") as f:
    f.write(f"$output_name MMLU: {accuracy:.4f} ({correct}/{total})\n")

print(f"Saved to $RESULTS_FILE")
EOF
}

eval_existing_adapter() {
    local model_path="$1"
    local output_name="${2:-$(basename "$model_path")}"
    require_adapter_dir "$model_path"
    run_mmlu_eval "$model_path" "$output_name" "$NUM_EVAL_SAMPLES"
}

# Evaluate base model (no training)
eval_base() {
    echo "=== Evaluating base model on MMLU ==="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" - <<EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print(f"Loading base model: $MODEL_ID")
model = AutoModelForCausalLM.from_pretrained(
    "$MODEL_ID",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("$MODEL_ID")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Load MMLU
ds = load_dataset("cais/mmlu", "all", split="test")
if $NUM_EVAL_SAMPLES > 0:
    ds = ds.select(range($NUM_EVAL_SAMPLES))
print(f"Test samples: {len(ds)}")

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
    if pred_idx == row["answer"]:
        correct += 1
    total += 1
    
    if (i + 1) % 200 == 0:
        print(f"  Progress: {i+1}/{total}, Accuracy: {correct/total:.4f}")

accuracy = correct / total
print(f"\n=== Base Model ===")
print(f"MMLU Accuracy ({total} samples): {accuracy:.4f}")
print(f"Correct: {correct}/{total}")

with open("$RESULTS_FILE", "a") as f:
    f.write(f"Base Model MMLU: {accuracy:.4f} ({correct}/{total})\n")
EOF
}

print_help() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  jora_oer         - Train JORA oer_softmax and evaluate"
    echo "  jora_none        - Train JORA magnitude=none and evaluate"
    echo "  lora_r4          - Train LoRA r=4 and evaluate"
    echo "  base             - Evaluate base model only"
    echo "  all              - Train all adapters and run full comparison"
    echo "  eval_jora_oer    - Evaluate existing \$WORKDIR/out_jora_oer"
    echo "  eval_jora_none   - Evaluate existing \$WORKDIR/out_jora_none"
    echo "  eval_lora_r4     - Evaluate existing \$WORKDIR/out_lora_r4"
    echo "  eval_all         - Evaluate base + existing adapters in \$WORKDIR"
    echo "  eval_existing    - Evaluate an existing adapter path"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_ID         - Model (default: facebook/opt-350m)"
    echo "  MAX_STEPS        - Training steps (default: 2000)"
    echo "  NUM_EVAL_SAMPLES - MMLU samples (0=full, default: 0)"
    echo "  WORKDIR          - Output directory"
}

case "${1:-}" in
    jora_oer)
        print_summary
        train_jora "$WORKDIR/out_jora_oer" "oer_softmax"
        run_mmlu_eval "$WORKDIR/out_jora_oer" "JORA-oer_softmax" "$NUM_EVAL_SAMPLES"
        ;;
    jora_none)
        print_summary
        train_jora "$WORKDIR/out_jora_none" "none"
        run_mmlu_eval "$WORKDIR/out_jora_none" "JORA-none" "$NUM_EVAL_SAMPLES"
        ;;
    lora_r4)
        print_summary
        train_lora "$WORKDIR/out_lora_r4" 4
        run_mmlu_eval "$WORKDIR/out_lora_r4" "LoRA-r4" "$NUM_EVAL_SAMPLES"
        ;;
    base)
        eval_base
        ;;
    all)
        print_summary
        rm -f "$RESULTS_FILE"
        eval_base
        train_jora "$WORKDIR/out_jora_oer" "oer_softmax"
        run_mmlu_eval "$WORKDIR/out_jora_oer" "JORA-oer_softmax" "$NUM_EVAL_SAMPLES"
        train_jora "$WORKDIR/out_jora_none" "none"
        run_mmlu_eval "$WORKDIR/out_jora_none" "JORA-none" "$NUM_EVAL_SAMPLES"
        train_lora "$WORKDIR/out_lora_r4" 4
        run_mmlu_eval "$WORKDIR/out_lora_r4" "LoRA-r4" "$NUM_EVAL_SAMPLES"
        
        echo ""
        echo "=== FINAL RESULTS ==="
        cat "$RESULTS_FILE"
        ;;
    eval_jora_oer)
        print_summary
        eval_existing_adapter "$WORKDIR/out_jora_oer" "JORA-oer_softmax"
        ;;
    eval_jora_none)
        print_summary
        eval_existing_adapter "$WORKDIR/out_jora_none" "JORA-none"
        ;;
    eval_lora_r4)
        print_summary
        eval_existing_adapter "$WORKDIR/out_lora_r4" "LoRA-r4"
        ;;
    eval_all)
        print_summary
        rm -f "$RESULTS_FILE"
        eval_base
        eval_existing_adapter "$WORKDIR/out_jora_oer" "JORA-oer_softmax"
        eval_existing_adapter "$WORKDIR/out_jora_none" "JORA-none"
        eval_existing_adapter "$WORKDIR/out_lora_r4" "LoRA-r4"

        echo ""
        echo "=== FINAL RESULTS ==="
        cat "$RESULTS_FILE"
        ;;
    eval_existing)
        if [ -z "${2:-}" ]; then
            echo "ERROR: provide adapter path"
            exit 1
        fi
        print_summary
        eval_existing_adapter "$2" "${3:-$(basename "$2")}"
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
