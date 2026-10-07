# Copyright 2026 Ashutosh Adhikari and Mirella Lapata.
# Adapted from LLM-JEPA (https://github.com/rbalestr-lab/llm-jepa), Apache-2.0.
# Modified: PEARL training entry point for Qwen-VL models.
"""Train a VLM with PEARL.

Launch with torchrun, e.g.::

    torchrun --nproc_per_node=4 -m pearl.train --dataset lvr \\
        --train_file /path/to/meta_data_lvr_sft_stage1.json \\
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --output_dir runs/pearl-lvr-7b
"""

import argparse
import os

import torch
from transformers import set_seed
from trl import SFTConfig

from .data import DATASETS, PEARLCollator, load_train_dataset
from .model import setup_model_and_processor
from .trainer import JEPA_LOSSES, PEARLTrainer

# Checkpoints saved per epoch in the paper runs.
SAVES_PER_EPOCH = {"lvr": 40, "thinkmorph": 10, "pixelreasoner": 1}

DEFAULT_DEEPSPEED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "configs", "ds_zero2.json")


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data / model
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--train_file", required=True,
                   help="lvr: path to the LVR metadata json; thinkmorph: ThinkMorph HF org or local root "
                        "holding the four task subsets; pixelreasoner: HF dataset name or local path.")
    p.add_argument("--image_root", default=None,
                   help="PixelReasoner only: directory holding the dataset's images/ folder.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output_dir", required=True)
    # PEARL
    p.add_argument("--lbd", type=float, default=0.2, help="Weight of L_JEPA + L_NextLat.")
    p.add_argument("--predictors", type=int, default=4, help="Number K of [PRED] tokens (max 10).")
    p.add_argument("--last_token", type=int, default=-2,
                   help="Embedding position relative to the end of each sequence (-2 = <|im_end|> for Qwen).")
    p.add_argument("--jepa_loss", choices=JEPA_LOSSES, default="cosine",
                   help="Distance for L_JEPA. The paper results were produced with 'cosine'.")
    p.add_argument("--no_jepa", action="store_true", help="Ablation: drop L_JEPA.")
    p.add_argument("--no_nextlat", action="store_true", help="Ablation: drop L_NextLat.")
    # Optimisation
    p.add_argument("--lora_rank", type=int, default=64, help="LoRA rank; alpha is 2 * rank.")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device batch size.")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1, help="Overrides --num_epochs when > 0.")
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=None,
                   help="Defaults to the per-dataset checkpoint frequency used in the paper.")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--deepspeed", default=DEFAULT_DEEPSPEED, help="DeepSpeed config ('' to disable).")
    p.add_argument("--report_to", default="wandb")
    p.add_argument("--no_liger", action="store_true", help="Disable Liger kernels.")
    p.add_argument("--no_merge", action="store_true",
                   help="Do not merge LoRA weights into the base model at the end of training.")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", "pearl")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    set_seed(args.seed)

    train_dataset = load_train_dataset(args.dataset, args.train_file)
    if is_main_process():
        print(f"Loaded {len(train_dataset)} {args.dataset} examples")

    save_steps = args.save_steps
    if save_steps is None:
        steps_per_epoch = len(train_dataset) // (world_size * args.batch_size * args.grad_accum)
        save_steps = max(1, steps_per_epoch // SAVES_PER_EPOCH[args.dataset])

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        bf16=True,
        tf32=False,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
        deepspeed=args.deepspeed or None,
        remove_unused_columns=False,
        dataset_text_field="message_list",
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=args.seed,
        data_seed=args.seed,
        use_liger_kernel=not args.no_liger,
    )

    model, processor = setup_model_and_processor(
        args.model_name, lora_rank=args.lora_rank, nextlat=not args.no_nextlat)
    if is_main_process():
        model.print_trainable_parameters()

    trainer = PEARLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=PEARLCollator(processor, args.dataset, predictors=args.predictors,
                                    image_root=args.image_root),
        lbd=args.lbd,
        last_token=args.last_token,
        jepa_loss=args.jepa_loss,
        use_jepa=not args.no_jepa,
        nextlat=not args.no_nextlat,
    )
    trainer.train()

    if is_main_process():
        if args.no_merge:
            model.save_pretrained(output_dir)
        else:
            model.merge_and_unload().save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()
