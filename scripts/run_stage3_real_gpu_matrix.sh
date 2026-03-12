#!/usr/bin/env bash
# Stage 3 real-model GPU matrix verification for JORA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/torchrun}"

export PYTHONPATH="${REPO_ROOT}/src"
export HF_HOME="${HF_HOME:-/tmp/jora-stage3-real/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jora-stage3-real/hf_datasets}"
WORKDIR="${WORKDIR:-/tmp/jora-stage3-real}"

MODEL_ID="${MODEL_ID:-facebook/opt-125m}"
DATASET_NAME="${DATASET_NAME:-yahma/alpaca-cleaned}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:256]}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SINGLE_STEPS="${SINGLE_STEPS:-50}"
AUTO_STEPS="${AUTO_STEPS:-20}"
DDP_STEPS="${DDP_STEPS:-20}"
RESUME_STEPS="${RESUME_STEPS:-20}"
RESUME_TARGET_STEPS="${RESUME_TARGET_STEPS:-30}"
DDP_RESUME_STEPS="${DDP_RESUME_STEPS:-10}"
DDP_RESUME_TARGET_STEPS="${DDP_RESUME_TARGET_STEPS:-15}"

JORA_TARGETS="q_proj,k_proj,v_proj,out_proj"
JORA_S_L="${JORA_S_L:-8}"
JORA_S_R="${JORA_S_R:-8}"
JORA_K="${JORA_K:-4}"
JORA_SELECTION="${JORA_SELECTION:-topk_ema}"
JORA_MAGNITUDE="${JORA_MAGNITUDE:-oer_softmax}"
JORA_WARMUP="${JORA_WARMUP:-10}"

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
assert (output_dir / "adapter_model.safetensors").exists(), output_dir

base = AutoModelForCausalLM.from_pretrained(snapshot_path, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, output_dir)
tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
model.eval().cuda()
inputs = tokenizer("Instruction: Explain JORA briefly.\nOutput:", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs)
print(f"Validated: logits_shape={tuple(out.logits.shape)}")
PY
}

train_single() {
    local output_dir="$1"
    local max_steps="$2"
    local rotation_impl="$3"
    ensure_absent "$output_dir"

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
        --max_steps "$max_steps" \
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
        --jora_rotation_impl "$rotation_impl" \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude "$JORA_MAGNITUDE" \
        --jora_warmup_steps "$JORA_WARMUP"

    validate_adapter_gpu "$output_dir"
}

resume_single() {
    local output_dir="$WORKDIR/out_opt125m_resume"
    ensure_absent "$output_dir"

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
        --max_steps "$RESUME_STEPS" \
        --logging_steps 5 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 10 \
        --save_total_limit 2 \
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
        --jora_rotation_impl torch \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude "$JORA_MAGNITUDE" \
        --jora_warmup_steps "$JORA_WARMUP"

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
        --max_steps "$RESUME_TARGET_STEPS" \
        --logging_steps 5 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 10 \
        --save_total_limit 2 \
        --report_to none \
        --output_dir "$output_dir" \
        --resume_from_checkpoint "$output_dir/checkpoint-$RESUME_STEPS" \
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
        --jora_rotation_impl torch \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude "$JORA_MAGNITUDE" \
        --jora_warmup_steps "$JORA_WARMUP"

    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
state = json.loads(Path("$output_dir/checkpoint-$RESUME_TARGET_STEPS/trainer_state.json").read_text())
assert state["global_step"] == $RESUME_TARGET_STEPS, state
print(f"Validated: global_step={state['global_step']}, max_steps={state['max_steps']}")
PY

    validate_adapter_gpu "$output_dir"
}

train_ddp2() {
    local output_dir="$1"
    local max_steps="$2"
    ensure_absent "$output_dir"

    CUDA_VISIBLE_DEVICES=0,1 "$TORCHRUN_BIN" --standalone --nproc_per_node=2 examples/sft/train.py \
        --seed 1234 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$max_steps" \
        --logging_steps 5 \
        --eval_strategy no \
        --save_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --dataset_text_field text \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_jora True \
        --lora_target_modules "$JORA_TARGETS" \
        --jora_s_l "$JORA_S_L" \
        --jora_s_r "$JORA_S_R" \
        --jora_k "$JORA_K" \
        --jora_rotation_impl torch \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude "$JORA_MAGNITUDE" \
        --jora_warmup_steps "$JORA_WARMUP"

    validate_adapter_gpu "$output_dir"
}

resume_ddp2() {
    local output_dir="$WORKDIR/out_opt125m_ddp2_resume"
    ensure_absent "$output_dir"

    CUDA_VISIBLE_DEVICES=0,1 "$TORCHRUN_BIN" --standalone --nproc_per_node=2 examples/sft/train.py \
        --seed 1234 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$DDP_RESUME_STEPS" \
        --logging_steps 5 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 5 \
        --save_total_limit 2 \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --dataset_text_field text \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_jora True \
        --lora_target_modules "$JORA_TARGETS" \
        --jora_s_l "$JORA_S_L" \
        --jora_s_r "$JORA_S_R" \
        --jora_k "$JORA_K" \
        --jora_rotation_impl torch \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude "$JORA_MAGNITUDE" \
        --jora_warmup_steps "$JORA_WARMUP"

    CUDA_VISIBLE_DEVICES=0,1 "$TORCHRUN_BIN" --standalone --nproc_per_node=2 examples/sft/train.py \
        --seed 1234 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "$DATASET_SPLIT" \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$DDP_RESUME_TARGET_STEPS" \
        --logging_steps 5 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 5 \
        --save_total_limit 2 \
        --report_to none \
        --output_dir "$output_dir" \
        --resume_from_checkpoint "$output_dir/checkpoint-$DDP_RESUME_STEPS" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --dataset_text_field text \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_jora True \
        --lora_target_modules "$JORA_TARGETS" \
        --jora_s_l "$JORA_S_L" \
        --jora_s_r "$JORA_S_R" \
        --jora_k "$JORA_K" \
        --jora_rotation_impl torch \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude "$JORA_MAGNITUDE" \
        --jora_warmup_steps "$JORA_WARMUP"

    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
state = json.loads(Path("$output_dir/checkpoint-$DDP_RESUME_TARGET_STEPS/trainer_state.json").read_text())
assert state["global_step"] == $DDP_RESUME_TARGET_STEPS, state
print(f"Validated: global_step={state['global_step']}, max_steps={state['max_steps']}")
PY

    validate_adapter_gpu "$output_dir"
}

case "${1:-}" in
    single_torch)
        train_single "$WORKDIR/out_opt125m_sg${SINGLE_STEPS}" "$SINGLE_STEPS" torch
        ;;
    resume)
        resume_single
        ;;
    ddp2)
        train_ddp2 "$WORKDIR/out_opt125m_ddp2_${DDP_STEPS}" "$DDP_STEPS"
        ;;
    ddp2_resume)
        resume_ddp2
        ;;
    auto)
        train_single "$WORKDIR/out_opt125m_auto${AUTO_STEPS}" "$AUTO_STEPS" auto
        ;;
    all)
        train_single "$WORKDIR/out_opt125m_sg${SINGLE_STEPS}" "$SINGLE_STEPS" torch
        resume_single
        train_ddp2 "$WORKDIR/out_opt125m_ddp2_${DDP_STEPS}" "$DDP_STEPS"
        resume_ddp2
        train_single "$WORKDIR/out_opt125m_auto${AUTO_STEPS}" "$AUTO_STEPS" auto
        ;;
    *)
        echo "Usage: $0 [single_torch|resume|ddp2|ddp2_resume|auto|all]" >&2
        exit 1
        ;;
esac
