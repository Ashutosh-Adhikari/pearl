#!/usr/bin/env bash
# Evaluate one or more checkpoints on V*, MMVP and BLINK.
# Usage: VSTAR_DIR=data/vstar_bench MMVP_DIR=data/MMVP bash scripts/eval.sh runs/pearl-lvr-Qwen2.5-VL-7B-Instruct/checkpoint-2268
set -euo pipefail
python eval/evaluate.py --checkpoints "$@" \
  --vstar_dir "${VSTAR_DIR:-data/vstar_bench}" --mmvp_dir "${MMVP_DIR:-data/MMVP}" \
  --preset "${PRESET:-default}" --output_dir "${RESULTS_DIR:-results}"
