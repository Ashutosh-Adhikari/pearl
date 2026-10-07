#!/usr/bin/env bash
# PEARL on ThinkMorph (multiple tool types, single call). Settings of the paper's ThinkMorph runs.
# Usage: bash scripts/train_thinkmorph.sh   (THINKMORPH=<HF org or local root>, default ThinkMorph)
set -euo pipefail
NGPUS=${NGPUS:-4}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-runs/pearl-thinkmorph-$(basename ${MODEL})}

torchrun --nproc_per_node=${NGPUS} -m pearl.train \
  --dataset thinkmorph --train_file "${THINKMORPH:-ThinkMorph}" \
  --model_name ${MODEL} --output_dir ${OUTPUT_DIR} \
  --lbd 0.2 --predictors 4 --last_token -2 --jepa_loss cosine \
  --lora_rank 64 --batch_size 4 --grad_accum 4 --learning_rate 1e-5 --num_epochs 4 \
  --lr_scheduler_type cosine --warmup_ratio 0.05 --weight_decay 0.1 "$@"
