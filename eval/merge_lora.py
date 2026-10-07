# Copyright 2026 Ashutosh Adhikari and Mirella Lapata.
# SPDX-License-Identifier: Apache-2.0
"""Merge a PEARL LoRA checkpoint (``checkpoint-N``) into its base model.

    python eval/merge_lora.py --checkpoint runs/pearl-lvr-7b/checkpoint-2268 --output_dir merged/pearl-lvr-7b

PEARL needs no extra modules at inference: the predictor tokens and the
NextLat dynamics head are only used during training, so the merged model is a
plain Qwen-VL model.
"""

import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pearl.model import vlm_class  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    with open(os.path.join(args.checkpoint, "adapter_config.json")) as f:
        base = json.load(f)["base_model_name_or_path"]
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = vlm_class(base).from_pretrained(base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if model.get_input_embeddings().weight.shape[0] < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(model, args.checkpoint).merge_and_unload()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
