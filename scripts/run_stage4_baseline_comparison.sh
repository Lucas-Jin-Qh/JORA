#!/usr/bin/env bash
# Stage 4: Baseline comparison matrix (JORA vs LoRA vs optional OFT/DoRA)
# Fixed settings: facebook/opt-125m, yahma/alpaca-cleaned, same split, same max_steps, same bf16/checkpointing/batch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/torchrun}"

export PYTHONPATH="${REPO_ROOT}/src"
export HF_HOME="${HF_HOME:-/tmp/jora-stage4-baseline/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jora-stage4-baseline/hf_datasets}"
WORKDIR="${WORKDIR:-/tmp/jora-stage4-baseline}"

MODEL_ID="${MODEL_ID:-facebook/opt-125m}"
DATASET_NAME="${DATASET_NAME:-yahma/alpaca-cleaned}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:1024]}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_STEPS="${MAX_STEPS:-100}"

# JORA config (winner from stage4)
JORA_TARGETS="q_proj,k_proj,v_proj,out_proj"
JORA_S_L="${JORA_S_L:-16}"
JORA_S_R="${JORA_S_R:-16}"
JORA_K="${JORA_K:-8}"
JORA_SELECTION="${JORA_SELECTION:-topk_ema}"
JORA_MAGNITUDE="${JORA_MAGNITUDE:-oer_softmax}"
JORA_WARMUP="${JORA_WARMUP:-10}"
JORA_OER_TEMP="${JORA_OER_TEMP:-1.0}"
JORA_LR_THETA="${JORA_LR_THETA:-}"
JORA_LR_CORE="${JORA_LR_CORE:-}"

# LoRA config for comparison
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,out_proj}"

ensure_absent() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Refusing to overwrite existing path: $path" >&2
        exit 1
    fi
}

resolve_model_snapshot() {
    local model_cache_id
    model_cache_id="models--${MODEL_ID//\//--}"
    find -L "$HF_HOME/hub/${model_cache_id}/snapshots" -maxdepth 2 -name config.json -printf '%h\n' | head -n 1
}

validate_adapter_gpu() {
    local output_dir="$1"
    local peft_type="$2"
    local snapshot_path
    snapshot_path="$(resolve_model_snapshot)"
    if [ -z "$snapshot_path" ]; then
        echo "Could not resolve local model snapshot under $HF_HOME" >&2
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<PY
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

snapshot_path = Path("$snapshot_path")
output_dir = Path("$output_dir")
assert (output_dir / "adapter_config.json").exists(), output_dir

base = AutoModelForCausalLM.from_pretrained(snapshot_path, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, output_dir)
tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
model.eval().cuda()
inputs = tokenizer("Instruction: Explain JORA briefly.\nOutput:", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs)
print(f"[${peft_type}] Validated: logits_shape={tuple(out.logits.shape)}")
PY
}

# JORA baseline (winner: S_L=16, S_R=16, k=8, rotation_impl=auto)
train_jora() {
    local output_dir="$1"
    ensure_absent "$output_dir"

    echo "=== Training JORA (S_L=$JORA_S_L, S_R=$JORA_S_R, k=$JORA_K) ==="
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
        --logging_steps 5 \
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
        --jora_magnitude "$JORA_MAGNITUDE" \
        --jora_warmup_steps "$JORA_WARMUP" \
        --jora_oer_temperature "${JORA_OER_TEMP:-1.0}" \
        $([ -n "$JORA_LR_THETA" ] && echo "--jora_lr_theta $JORA_LR_THETA") \
        $([ -n "$JORA_LR_CORE" ] && echo "--jora_lr_core $JORA_LR_CORE")

    validate_adapter_gpu "$output_dir" "JORA"
}

# LoRA baseline (r matched to JORA)
train_lora() {
    local output_dir="$1"
    local r="$2"
    ensure_absent "$output_dir"

    echo "=== Training LoRA (r=$r, alpha=$LORA_ALPHA) ==="
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
        --logging_steps 5 \
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
        --lora_target_modules "$LORA_TARGETS" \
        --lora_r "$r" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT"

    validate_adapter_gpu "$output_dir" "LoRA-r${r}"
}

# OFT baseline (r specifies rank; block_size auto-computed from hidden_dim/r)
train_oft() {
    local output_dir="$1"
    local oft_r="${2:-8}"
    ensure_absent "$output_dir"

    echo "=== Training OFT (r=$oft_r) ==="
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
        --logging_steps 5 \
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
        --use_peft_oft True \
        --lora_target_modules "$LORA_TARGETS" \
        --oft_r "$oft_r"

    validate_adapter_gpu "$output_dir" "OFT-r${oft_r}"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "       BASELINE COMPARISON SUMMARY"
    echo "========================================"
    echo ""
    echo "Settings:"
    echo "  Model:     $MODEL_ID"
    echo "  Dataset:   $DATASET_NAME ($DATASET_SPLIT)"
    echo "  Max steps: $MAX_STEPS"
    echo "  LR:        $LEARNING_RATE"
    echo "  BF16:      True"
    echo "  Gradient checkpointing: True"
    echo ""
    echo "Results:"
    echo ""
}

case "${1:-}" in
    jora)
        print_summary
        train_jora "$WORKDIR/out_jora_16_16_8"
        ;;
    jora_t1)
        print_summary
        JORA_OER_TEMP=1.0 train_jora "$WORKDIR/out_jora_t1"
        ;;
    jora_t2)
        print_summary
        JORA_OER_TEMP=2.0 train_jora "$WORKDIR/out_jora_t2"
        ;;
    jora_no_mag)
        print_summary
        JORA_MAGNITUDE=none train_jora "$WORKDIR/out_jora_no_mag"
        ;;
    jora_t2_lr)
        print_summary
        JORA_OER_TEMP=2.0 JORA_LR_THETA=0.05 JORA_LR_CORE=0.01 train_jora "$WORKDIR/out_jora_t2_lr"
        ;;
    lora)
        print_summary
        train_lora "$WORKDIR/out_lora_r${LORA_R}" "$LORA_R"
        ;;
    lora_r4)
        print_summary
        train_lora "$WORKDIR/out_lora_r4" 4
        ;;
    lora_r1)
        print_summary
        train_lora "$WORKDIR/out_lora_r1" 1
        ;;
    lora_r8)
        print_summary
        train_lora "$WORKDIR/out_lora_r8" 8
        ;;
    lora_r16)
        print_summary
        train_lora "$WORKDIR/out_lora_r16" 16
        ;;
    oft)
        print_summary
        train_oft "$WORKDIR/out_oft_r8" 8
        ;;
    all)
        print_summary
        echo "Running: JORA 16/16/8 vs LoRA r=4/8/16 vs OFT r=8"
        echo ""
        train_jora "$WORKDIR/out_jora_16_16_8"
        echo ""
        train_lora "$WORKDIR/out_lora_r4" 4
        echo ""
        train_lora "$WORKDIR/out_lora_r8" 8
        echo ""
        train_lora "$WORKDIR/out_lora_r16" 16
        echo ""
        train_oft "$WORKDIR/out_oft_r8" 8
        ;;
    *)
        echo "Usage: $0 [jora|lora|lora_r4|lora_r8|lora_r16|oft|all]"
        echo ""
        echo "Comparison matrix:"
        echo "  jora      - JORA S_L=16 S_R=16 k=8 (baseline winner)"
        echo "  lora      - LoRA r=$LORA_R (configurable)"
        echo "  lora_r4   - LoRA r=4"
        echo "  lora_r8   - LoRA r=8"
        echo "  lora_r16  - LoRA r=16"
        echo "  oft       - OFT r=8"
        echo "  all       - Run all comparisons"
        echo ""
        echo "Environment variables:"
        echo "  WORKDIR=/tmp/jora-baseline"
        echo "  DATASET_SPLIT=train[:1024]"
        echo "  MAX_STEPS=100"
        echo "  LORA_R=8"
        exit 1
        ;;
esac
