#!/usr/bin/env bash
# Stage 2 GPU matrix verification script for JORA
# This script validates single/multi-GPU training, checkpoint resume, and DDP compatibility

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Default values
WORKDIR="${WORKDIR:-/tmp/jora-gpu-matrix}"
STEPS="${STEPS:-20}"
GPUS="${GPUS:-1}"
RESUME="${RESUME:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"

# JORA config (shared across runs)
SEED=1234
MODEL_PATH="$WORKDIR/model"
DATASET_PATH="$WORKDIR/dataset"
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,out_proj"
JORA_S_L=4
JORA_S_R=4
JORA_K=2
JORA_ROTATION_IMPL="torch"
JORA_SELECTION_TYPE="topk_ema"
JORA_MAGNITUDE="oer_softmax"
JORA_WARMUP_STEPS=2
LEARNING_RATE=0.005
MAX_LENGTH=32

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate peft-jora
export PYTHONPATH="$REPO_ROOT/src"

# Build offline model + dataset if not exists
if [ "$SKIP_BUILD" = "false" ]; then
    echo "=== Building offline tiny model + dataset ==="
    python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from jora_cli_smoke import build_model_bundle, build_dataset

workdir = Path("/tmp/jora-gpu-matrix")
workdir.mkdir(parents=True, exist_ok=True)
build_model_bundle(workdir / "model")
build_dataset(workdir / "dataset")
print("Built:", workdir)
PY
fi

run_single_gpu() {
    local output_dir="$1"
    local max_steps="$2"
    echo ""
    echo "=== Single GPU: $max_steps steps ==="
    echo "Output: $output_dir"

    rm -rf "$output_dir"

    CUDA_VISIBLE_DEVICES=0 python examples/sft/train.py \
        --seed $SEED \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_name "$DATASET_PATH" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits train,test \
        --torch_dtype float32 \
        --max_steps $max_steps \
        --logging_steps 1 \
        --eval_strategy no \
        --save_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing False \
        --use_peft_jora True \
        --lora_target_modules $LORA_TARGET_MODULES \
        --jora_s_l $JORA_S_L \
        --jora_s_r $JORA_S_R \
        --jora_k $JORA_K \
        --jora_rotation_impl $JORA_ROTATION_IMPL \
        --jora_selection_type $JORA_SELECTION_TYPE \
        --jora_magnitude $JORA_MAGNITUDE \
        --jora_warmup_steps $JORA_WARMUP_STEPS

    # Validate adapter
    python - <<PY
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
out_dir = Path("$output_dir")
assert (out_dir / "adapter_config.json").exists()
assert (out_dir / "adapter_model.safetensors").exists()
base = AutoModelForCausalLM.from_pretrained(Path("$MODEL_PATH"))
model = PeftModel.from_pretrained(base, out_dir)
tok = AutoTokenizer.from_pretrained(Path("$MODEL_PATH"))
model.eval()
inputs = tok("Instruction : Write a short response about JORA Output :", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
print(f"Validated: logits_shape={tuple(out.logits.shape)}")
PY
    echo "=== Single GPU $max_steps steps: PASSED ==="
}

run_resume() {
    local output_dir="$WORKDIR/out_resume"
    local init_steps=8
    local resume_steps=12

    echo ""
    echo "=== Resume: $init_steps -> $resume_steps ==="
    echo "Output: $output_dir"

    rm -rf "$output_dir"

    # Phase 1: train to init_steps
    CUDA_VISIBLE_DEVICES=0 python examples/sft/train.py \
        --seed $SEED \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_name "$DATASET_PATH" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits train,test \
        --torch_dtype float32 \
        --max_steps $init_steps \
        --logging_steps 2 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 4 \
        --save_total_limit 2 \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing False \
        --use_peft_jora True \
        --lora_target_modules $LORA_TARGET_MODULES \
        --jora_s_l $JORA_S_L \
        --jora_s_r $JORA_S_R \
        --jora_k $JORA_K \
        --jora_rotation_impl $JORA_ROTATION_IMPL \
        --jora_selection_type $JORA_SELECTION_TYPE \
        --jora_magnitude $JORA_MAGNITUDE \
        --jora_warmup_steps $JORA_WARMUP_STEPS

    # Phase 2: resume to resume_steps
    CUDA_VISIBLE_DEVICES=0 python examples/sft/train.py \
        --seed $SEED \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_name "$DATASET_PATH" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits train,test \
        --torch_dtype float32 \
        --max_steps $resume_steps \
        --logging_steps 2 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 4 \
        --save_total_limit 2 \
        --report_to none \
        --output_dir "$output_dir" \
        --resume_from_checkpoint "$output_dir/checkpoint-$init_steps" \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing False \
        --use_peft_jora True \
        --lora_target_modules $LORA_TARGET_MODULES \
        --jora_s_l $JORA_S_L \
        --jora_s_r $JORA_S_R \
        --jora_k $JORA_K \
        --jora_rotation_impl $JORA_ROTATION_IMPL \
        --jora_selection_type $JORA_SELECTION_TYPE \
        --jora_magnitude $JORA_MAGNITUDE \
        --jora_warmup_steps $JORA_WARMUP_STEPS

    # Validate global_step continuity
    python - <<PY
import json
from pathlib import Path
state = json.loads(Path("$output_dir/checkpoint-$resume_steps/trainer_state.json").read_text())
assert state["global_step"] == $resume_steps, f"Expected global_step=$resume_steps, got {state['global_step']}"
print(f"Validated: global_step={state['global_step']}, max_steps={state['max_steps']}")
PY

    # Validate adapter
    python - <<PY
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
out_dir = Path("$output_dir")
assert (out_dir / "adapter_config.json").exists()
assert (out_dir / "adapter_model.safetensors").exists()
base = AutoModelForCausalLM.from_pretrained(Path("$MODEL_PATH"))
model = PeftModel.from_pretrained(base, out_dir)
tok = AutoTokenizer.from_pretrained(Path("$MODEL_PATH"))
model.eval()
inputs = tok("Instruction : Write a short response about JORA Output :", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
print(f"Validated: logits_shape={tuple(out.logits.shape)}")
PY
    echo "=== Resume $init_steps -> $resume_steps: PASSED ==="
}

run_ddp() {
    local num_gpus="$1"
    local output_dir="$2"
    local max_steps="$3"

    echo ""
    echo "=== DDP $num_gpus GPUs: $max_steps steps ==="
    echo "Output: $output_dir"

    rm -rf "$output_dir"

    local gpu_ids=$(seq -s, 0 $((num_gpus - 1)))
    CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --standalone --nproc_per_node=$num_gpus examples/sft/train.py \
        --seed $SEED \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_name "$DATASET_PATH" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits train,test \
        --torch_dtype float32 \
        --max_steps $max_steps \
        --logging_steps 1 \
        --eval_strategy no \
        --save_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing False \
        --use_peft_jora True \
        --lora_target_modules $LORA_TARGET_MODULES \
        --jora_s_l $JORA_S_L \
        --jora_s_r $JORA_S_R \
        --jora_k $JORA_K \
        --jora_rotation_impl $JORA_ROTATION_IMPL \
        --jora_selection_type $JORA_SELECTION_TYPE \
        --jora_magnitude $JORA_MAGNITUDE \
        --jora_warmup_steps $JORA_WARMUP_STEPS

    # Validate adapter
    python - <<PY
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
out_dir = Path("$output_dir")
assert (out_dir / "adapter_config.json").exists()
assert (out_dir / "adapter_model.safetensors").exists()
base = AutoModelForCausalLM.from_pretrained(Path("$MODEL_PATH"))
model = PeftModel.from_pretrained(base, out_dir)
tok = AutoTokenizer.from_pretrained(Path("$MODEL_PATH"))
model.eval()
inputs = tok("Instruction : Write a short response about JORA Output :", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
print(f"Validated: logits_shape={tuple(out.logits.shape)}")
PY
    echo "=== DDP $num_gpus GPUs $max_steps steps: PASSED ==="
}

# Parse arguments
case "$1" in
    single)
        run_single_gpu "$WORKDIR/out_sg_${STEPS}" "$STEPS"
        ;;
    resume)
        run_resume
        ;;
    ddp2)
        run_ddp 2 "$WORKDIR/out_ddp2_${STEPS}" "$STEPS"
        ;;
    ddp3)
        run_ddp 3 "$WORKDIR/out_ddp3_${STEPS}" "$STEPS"
        ;;
    all)
        run_single_gpu "$WORKDIR/out_sg_${STEPS}" "$STEPS"
        if [ "$RESUME" = "true" ]; then
            run_resume
        fi
        run_ddp 2 "$WORKDIR/out_ddp2_${STEPS}" "$STEPS"
        run_ddp 3 "$WORKDIR/out_ddp3_${STEPS}" "$STEPS"
        ;;
    *)
        echo "Usage: $0 [single|resume|ddp2|ddp3|all] [STEPS=20] [GPUS=1] [RESUME=false]"
        echo ""
        echo "Examples:"
        echo "  $0 single 20           # Single GPU 20 steps"
        echo "  $0 resume              # Resume 8 -> 12 steps"
        echo "  $0 ddp2 20             # 2-GPU DDP 20 steps"
        echo "  $0 ddp3 20             # 3-GPU DDP 20 steps"
        echo "  $0 all 20 RESUME=true  # Run all with resume"
        exit 1
        ;;
esac
