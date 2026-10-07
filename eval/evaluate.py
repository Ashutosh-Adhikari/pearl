# Copyright 2026 Ashutosh Adhikari and Mirella Lapata.
# Adapted from the LVR evaluation code (https://github.com/VincentLeebang/lvr), Apache-2.0.
# Modified: PEARL / LoRA checkpoint loading, Qwen3-VL support, prompt presets,
# command-line configuration.
"""Evaluate PEARL checkpoints on V*, MMVP and BLINK.

Example::

    python eval/evaluate.py --checkpoints runs/pearl-lvr-7b \\
        --vstar_dir data/vstar_bench --mmvp_dir data/MMVP --output_dir results/

``--checkpoints`` accepts merged models, LoRA checkpoint directories
(``checkpoint-N`` with an ``adapter_config.json``) or Hub model ids.
Predictions are written to ``<output_dir>/<run_name>/<benchmark>.json`` and
re-scored instead of regenerated when the file already exists.
"""

import argparse
import csv
import json
import os
import string
import sys

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pearl.data import SPECIAL_TOKENS  # noqa: E402
from pearl.model import vlm_class  # noqa: E402

LETTER = "\nAnswer with the option's letter from the given choices directly."
LETTER_TAGS = "\nAnswer with the option's letter from the given choices directly between <answer></answer> tags."
BOXED = "\nAnswer with the correct option's letter directly within \\boxed{}."

BLINK_SUBSETS = ["Counting", "IQ_Test", "Jigsaw", "Relative_Reflectance", "Spatial_Relation"]

# Prompt settings. `answer_prefix` appends "<answer>" after the generation prompt;
# `reject_multi` scores answers containing "AB" as wrong (models listing every option).
PRESETS = {
    "default": dict(instruction={"vstar": "", "mmvp": "", "blink": LETTER}, answer_prefix=True),
    "instruct": dict(instruction={"vstar": LETTER, "mmvp": LETTER, "blink": LETTER}, answer_prefix=True),
    "instruct_no_prefix": dict(instruction={"vstar": "", "mmvp": LETTER, "blink": LETTER}, answer_prefix=False),
    "thinkmorph": dict(instruction={"vstar": LETTER_TAGS, "mmvp": LETTER_TAGS, "blink": LETTER_TAGS},
                       answer_prefix=True, reject_multi=True),
    "boxed": dict(instruction={"vstar": BOXED, "mmvp": BOXED, "blink": BOXED}, answer_prefix=False,
                  blink_subsets=BLINK_SUBSETS + ["Object_Localization"]),
}


def is_correct(response, ground_truth, reject_multi=False):
    answer = response.split("<answer>")[-1].split("</answer")[0].strip()
    if reject_multi and "AB" in answer:
        return False
    if ground_truth in answer:
        return True
    if " " in answer:
        answer = answer.split(" ")[0]
    if len(answer) > 1:
        answer = answer[0]
    return answer == ground_truth


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(checkpoint):
    adapter_config = os.path.join(checkpoint, "adapter_config.json")
    if os.path.exists(adapter_config):
        with open(adapter_config) as f:
            base = json.load(f)["base_model_name_or_path"]
    else:
        base = checkpoint
    processor = AutoProcessor.from_pretrained(checkpoint)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    # transformers attaches the LoRA adapter automatically for adapter checkpoints.
    model = vlm_class(base).from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2", low_cpu_mem_usage=True)
    model.eval()
    return model, processor


def generate(model, processor, images, question, answer_prefix, max_new_tokens=4):
    images = images if isinstance(images, list) else [images]
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
                 + [{"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if answer_prefix:
        prompt += "<answer>"
    image_inputs, video_inputs = process_vision_info(messages)
    try:
        inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
    except ValueError as e:
        if "token count" in str(e) or "Mismatch" in str(e) or "truncat" in str(e).lower():
            print(f"[skip] image too large: {e}")
            return "[SKIPPED]"
        raise
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def load_vstar(args):
    data = []
    for d in load_dataset("craigwu/vstar_bench")["test"]:
        data.append({"id": d["question_id"], "image": os.path.join(args.vstar_dir, d["image"]),
                     "query": d["text"], "label": d["label"], "category": d["category"]})
    return data


def load_mmvp(args):
    data = []
    with open(os.path.join(args.mmvp_dir, "Questions.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["Index"])
            query = (row["Question"] + "\nOptions:\n" + row["Options"]).replace("(a)", "A.").replace("(b)", "B.")
            label = row["Correct Answer"]
            if label in ("(a)", "(b)"):
                label = label.strip().upper()[1]
            data.append({"id": idx, "image": os.path.join(args.mmvp_dir, "MMVP_Images", f"{idx}.jpg"),
                         "query": query, "label": label, "category": "mmvp"})
    return data


def load_blink(args, subsets):
    data = []
    for subset in subsets:
        for d in load_dataset("BLINK-Benchmark/BLINK", subset)["val"]:
            options = "".join(f"{l}. {c}\n" for l, c in zip(string.ascii_uppercase, d["choices"]))
            answer = d["answer"][1] if len(d["answer"]) > 1 else d["answer"][0]
            images = [d[k] for k in ("image_1", "image_2", "image_3", "image_4") if d.get(k) is not None]
            data.append({"id": d["idx"], "image": images, "query": d["question"] + "\nOptions:\n" + options,
                         "label": answer.upper(), "category": subset})
    return data


def evaluate(model, processor, name, data, out_file, preset):
    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
    else:
        results = []
        for d in tqdm(data, desc=name):
            prediction = generate(model, processor, d["image"], d["query"] + preset["instruction"][name],
                                  preset["answer_prefix"])
            results.append({"id": d["id"], "prediction": [prediction], "label": d["label"],
                            "category": d["category"]})
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    by_category = {}
    for r in results:
        correct = is_correct(r["prediction"][0], r["label"], preset.get("reject_multi", False))
        c = by_category.setdefault(r["category"], [0, 0])
        c[0] += int(correct)
        c[1] += 1
    total = sum(c[1] for c in by_category.values())
    accuracy = sum(c[0] for c in by_category.values()) / max(total, 1)
    summary = {"accuracy": accuracy, "n": total,
               "per_category": {k: c[0] / c[1] for k, c in by_category.items()}}
    print(f"{name}: {accuracy * 100:.2f} (n={total})")
    if len(by_category) > 1:
        for k, v in summary["per_category"].items():
            print(f"  {k}: {v * 100:.2f}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--benchmarks", nargs="+", default=["mmvp", "vstar", "blink"], choices=["mmvp", "vstar", "blink"])
    p.add_argument("--vstar_dir", help="Local copy of the craigwu/vstar_bench dataset repo (holds the images).")
    p.add_argument("--mmvp_dir", help="Local copy of MMVP (Questions.csv and MMVP_Images/).")
    p.add_argument("--blink_subsets", nargs="+", default=None, help="Overrides the preset's BLINK subsets.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="default", help="Prompt setting, see README.")
    p.add_argument("--output_dir", default="results")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    subsets = args.blink_subsets or preset.get("blink_subsets", BLINK_SUBSETS)
    loaders = {"vstar": lambda: load_vstar(args), "mmvp": lambda: load_mmvp(args),
               "blink": lambda: load_blink(args, subsets)}

    for checkpoint in args.checkpoints:
        run_name = checkpoint.strip("/").replace("/", "_") + f"_{args.preset}"
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== {checkpoint} (preset: {args.preset}) ===")
        model, processor = load_model(checkpoint)
        summary = {}
        for name in args.benchmarks:
            summary[name] = evaluate(model, processor, name, loaders[name](),
                                     os.path.join(run_dir, f"{name}.json"), preset)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
