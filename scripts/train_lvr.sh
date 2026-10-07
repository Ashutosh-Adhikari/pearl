#!/usr/bin/env bash
# PEARL on LVR data (single tool call: crop). Settings of the paper's LVR runs.
# Usage: LVR_META=/path/to/meta_data_lvr_sft_stage1.json bash scripts/train_lvr.sh
set -euo pipefail
NGPUS=${NGPUS:-4}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-runs/pearl-lvr-$(basename ${MODEL})}

torchrun --nproc_per_node=${NGPUS} -m pearl.train \
  --dataset lvr --train_file "${LVR_META:?set LVR_META to the LVR metadata json}" \
  --model_name ${MODEL} --output_dir ${OUTPUT_DIR} \
  --lbd 0.2 --predictors 4 --last_token -2 --jepa_loss cosine \
  --lora_rank 64 --batch_size 2 --grad_accum 8 --learning_rate 1e-5 --num_epochs 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.05 --weight_decay 0.1 "$@"
