#!/usr/bin/env bash
# PEARL on PixelReasoner SFT data (single tool type, multiple calls). Settings of the paper's PixelReasoner runs.
# Usage: PR_IMAGES=/path/to/PixelReasoner-SFT-Data bash scripts/train_pixelreasoner.sh
set -euo pipefail
NGPUS=${NGPUS:-4}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-runs/pearl-pixelreasoner-$(basename ${MODEL})}

torchrun --nproc_per_node=${NGPUS} -m pearl.train \
  --dataset pixelreasoner --train_file TIGER-Lab/PixelReasoner-SFT-Data \
  --image_root "${PR_IMAGES:?set PR_IMAGES to the local PixelReasoner-SFT-Data snapshot}" \
  --model_name ${MODEL} --output_dir ${OUTPUT_DIR} \
  --lbd 0.2 --predictors 4 --last_token -2 --jepa_loss cosine \
  --lora_rank 64 --batch_size 2 --grad_accum 2 --learning_rate 1e-5 --num_epochs 4 \
  --lr_scheduler_type cosine --warmup_ratio 0.3 --weight_decay 0.1 "$@"
