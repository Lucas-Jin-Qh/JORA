#!/usr/bin/env bash
# Launch ablation training experiments on free GPUs.
# abl_4mod_jora eval is already running on GPU0 (screen abl_4mod_jora_eval).
# GPU1 is free -> launch abl_oer (oer_softmax magnitude) on GPU1
# GPU0 also queued jobs: abl_warmup500, abl_lr_theta_003 (after eval finishes)
# GPU2 still busy -> abl_k32, abl_highlow, abl_lr_theta_005 queued for later

LOGDIR=/home/jqh/Workshop/JORA/formal_runs/ablation_seed42

# --- GPU1: abl_oer ---
screen -dmS abl_oer bash -c "
  export CUDA_VISIBLE_DEVICES=1
  export PYTHONPATH=/home/jqh/Workshop/JORA/src:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
  export HF_HOME=/home/jqh/Workshop/JORA/formal_runs/three_gpu_bf16/hf_home
  export HF_DATASETS_CACHE=/home/jqh/Workshop/JORA/formal_runs/three_gpu_bf16/hf_datasets
  export HF_HUB_DISABLE_XET=1
  export TOKENIZERS_PARALLELISM=false
  /home/jqh/miniconda3/envs/peft-jora/bin/python /home/jqh/Workshop/JORA/examples/sft/train.py \
    --seed 42 \
    --model_name_or_path /mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1 \
    --dataset_name yahma/alpaca-cleaned \
    --chat_template_format none \
    --add_special_tokens False \
    --append_concat_token False \
    --splits train \
    --torch_dtype bfloat16 \
    --bf16 True \
    --logging_steps 100 \
    --eval_strategy no \
    --save_strategy epoch \
    --report_to none \
    --output_dir $LOGDIR/abl_oer \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0001 \
    --max_length 512 \
    --dataset_text_field text \
    --use_cpu False \
    --gradient_checkpointing True \
    --use_reentrant False \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --save_total_limit 2 \
    --use_peft_jora True \
    --jora_core selective_diag \
    --jora_target_modules q_proj,o_proj \
    --jora_magnitude oer_softmax \
    --jora_t_stat 200 \
    --jora_pairs_freeze_after_warmup True \
    --jora_selection_type topk_ema \
    --jora_s_l 96 \
    --jora_s_r 96 \
    --jora_k 16 \
    --jora_lr_theta 0.001 \
    --jora_lr_core 0.0005 \
    2>&1 | tee $LOGDIR/abl_oer/train.log
"

echo "Launched abl_oer on GPU1 (screen: abl_oer)"
screen -ls
