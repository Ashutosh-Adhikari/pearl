# Copyright 2026 Ashutosh Adhikari and Mirella Lapata.
# Adapted from LLM-JEPA (https://github.com/rbalestr-lab/llm-jepa), Apache-2.0.
# Modified: Qwen2.5-VL / Qwen3-VL loading, NextLat dynamics head.
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoProcessor

from .data import SPECIAL_TOKENS
from .dynamics_head import DynamicsHead


def text_config(config):
    return getattr(config, "text_config", None) or config


def vlm_class(model_name):
    """Return the model class for a Qwen2.5-VL or Qwen3-VL checkpoint."""
    model_type = AutoConfig.from_pretrained(model_name).model_type
    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration  # transformers>=4.57
        return Qwen3VLForConditionalGeneration
    raise ValueError(f"Unsupported model type '{model_type}'; expected qwen2_5_vl or qwen3_vl")


def load_processor(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor, bool(new_tokens)


def setup_model_and_processor(model_name, lora_rank=64, nextlat=True):
    """Load the VLM, add the NextLat head and wrap everything with LoRA."""
    processor, added_tokens = load_processor(model_name)
    cls = vlm_class(model_name)
    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                  attn_implementation="flash_attention_2")
    if cls.__name__.startswith("Qwen2_5"):
        kwargs["use_cache"] = False
    model = cls.from_pretrained(model_name, **kwargs)

    if nextlat:
        model.dynamics_head = DynamicsHead(hidden_size=text_config(model.config).hidden_size)
    if added_tokens and text_config(model.config).vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.1,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    return model, processor
