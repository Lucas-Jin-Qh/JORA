#!/bin/bash
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

# Example script to run JORA training
# Usage: bash scripts/run_peft_jora.sh

export CUDA_VISIBLE_DEVICES=0

python examples/sft/train.py \
    --seed 42 \
    --model_name_or_path "microsoft/DialoGPT-small" \
    --dataset_name "timdettmers/openassistant-guanaco" \
    --chat_template_format "none" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train,test" \
    --max_seq_length 512 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --log_level "info" \
    --logging_strategy "steps" \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "epoch" \
    --bf16 False \
    --packing False \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --output_dir "outputs/jora_test" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --use_reentrant True \
    --dataset_text_field "text" \
    --use_peft_jora True \
    --lora_target_modules "c_attn,c_proj" \
    --jora_s_l 8 \
    --jora_s_r 8 \
    --jora_k 4 \
    --jora_rotation_param "cayley" \
    --jora_selection_type "topk_ema" \
    --jora_magnitude "ecd_tanh" \
    --report_to "none"
