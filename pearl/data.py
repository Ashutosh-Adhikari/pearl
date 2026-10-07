# Copyright 2026 Ashutosh Adhikari and Mirella Lapata.
# Adapted from LLM-JEPA (https://github.com/rbalestr-lab/llm-jepa), Apache-2.0.
# Modified: replaced text-only datasets with multimodal tool-use trajectories
# (LVR, ThinkMorph, PixelReasoner) and the PEARL input / trajectory views.
"""Datasets, chat-message construction and the PEARL data collator.

Every training example is turned into a multi-turn chat in which the assistant
turns form an expert tool-use *trajectory*. From that chat we build three views:

* the full conversation, trained with the usual next-token (VLM) loss;
* the input view (system + image-question) with K ``<|predictor_k|>`` tokens
  appended, whose last hidden state is the predicted trajectory embedding;
* the trajectory view (system + image + assistant/tool turns, without the
  question text), whose last hidden state is the JEPA target.
"""

import math
from copy import deepcopy

import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info

DATASETS = ("lvr", "thinkmorph", "pixelreasoner")

THINKMORPH_SUBSETS = ["Visual_Search", "Jigsaw_Assembly", "Chart_Refocus", "Spatial_Navigation"]

# Predictor tokens plus a few extra tokens inherited from LLM-JEPA. The list is
# kept as is so that the tokenizer (and hence released checkpoints) match the
# models reported in the paper.
SPECIAL_TOKENS = [f"<|predictor_{i}|>" for i in range(1, 11)] + [
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>",
]


def get_turn(role, text=None, image=None):
    turn = {"role": role, "content": []}
    if image:
        turn["content"].append({"type": "image", "image": image})
    if text:
        turn["content"].append({"type": "text", "text": text})
    return turn


# ---------------------------------------------------------------------------
# Per-dataset message construction
# ---------------------------------------------------------------------------

def get_messages_lvr(example):
    """LVR: question -> "focus on bbox" -> cropped region -> answer.

    The single crop of the LVR data is rendered as an explicit tool call
    (assistant names the normalised bounding box) and tool result (user turn
    holding the cropped image), followed by the answer.
    """
    messages = [get_turn("system", text="You are a helpful assistant.")]
    for msg in example["conversations"]:
        image = None
        turn = {"role": "user" if msg["from"] == "human" else "assistant", "content": []}
        if turn["role"] == "user":
            image = Image.open(example["image"])
            turn["content"].append({"type": "image", "image": image})
        text = (msg["value"].replace("<lvr>\n", "").replace("<image>\n", "")
                .replace("<answer>", "").replace("</answer>", ""))
        turn["content"].append({"type": "text", "text": text})
        messages.append(turn)

        if image:
            w, h = image.size
            x_min, y_min, x_max, y_max = example["bboxes"]
            assert x_min < x_max and y_min < y_max, f"Invalid bbox: {example['bboxes']}"
            xmin, ymin = int(math.floor(x_min * w)), int(math.floor(y_min * h))
            xmax, ymax = int(math.floor(x_max * w)), int(math.floor(y_max * h))
            assert xmin < xmax <= w and ymin < ymax <= h, f"Invalid bbox: {example['bboxes']} for image {(w, h)}"
            messages.append(get_turn(
                "assistant",
                text=f"Now I will focus on the relevant part of the image within the normalized bounding boxes: {example['bboxes']}."))
            messages.append({"role": "user", "content": [
                {"type": "image", "image": image.crop((xmin, ymin, xmax, ymax))},
                {"type": "text", "text": f"Here is the relevant cropped part of the image within the normalized bounding boxes: {example['bboxes']}."},
            ]})
    return messages


def get_messages_thinkmorph(example):
    """ThinkMorph: question -> thought -> manipulated image -> thought + answer."""
    return [
        get_turn("system", text="You are a helpful assistant.\n"),
        get_turn("user", text=example["question"], image=example["problem_image_0"]),
        get_turn("assistant", text=example["resoning_thought_0"]),
        get_turn("user", text="Here is the focused image.", image=example["reasoning_image_0"]),
        get_turn("assistant", text=example["resoning_thought_1"]),
    ]


def get_messages_pixelreasoner(example, image_root):
    """PixelReasoner: multi-turn trajectories with several sequential crops.

    Tool calls are stripped from assistant turns (PEARL never calls tools at
    inference); the tool outputs (cropped images) stay in the trajectory.
    """
    messages = []
    for i, msg in enumerate(example["message_list"]):
        is_system, is_first_user = i == 0, i == 1
        turn = {"role": msg["role"], "content": []}
        non_null = [{k: v for k, v in c.items() if v is not None} for c in msg["content"]]
        for content in non_null:
            for key, value in content.items():
                assert key in ("text", "image"), f"Unexpected content type: {key}"
                if key == "image":
                    turn["content"] = [{"type": "image", "image": f"{image_root}/{value}"}] + turn["content"]
                    continue
                if is_system:
                    value = value[: len("You are a helpful assistant.")]
                elif not is_first_user:
                    tool_idx = value.find("<tool_call>")
                    if tool_idx > -1:
                        value = value[:tool_idx]
                turn["content"].append({"type": "text", "text": value})
        messages.append(turn)
    return messages


def get_user_messages(messages, predictors=0):
    """Input view: system + first user turn, with K predictor tokens appended."""
    user_messages = deepcopy(messages[:2])
    assert user_messages[-1]["role"] == "user"
    pred_text = "".join(f"<|predictor_{k}|>" for k in range(predictors, 0, -1))
    user_messages[-1]["content"].append({"type": "text", "text": pred_text})
    return user_messages


def get_trajectory_messages(messages):
    """Trajectory view: the question text is dropped, the image is kept."""
    trajectory = deepcopy(messages)
    image = next((c for c in trajectory[1]["content"] if c["type"] == "image"), None)
    assert image is not None, f"First user turn has no image: {trajectory[1]}"
    trajectory[1]["content"] = [image]
    return trajectory


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _valid_bbox(bbox, image_size, max_ratio=200):
    w, h = image_size
    if min(h, w) == 0:
        return False
    xmin, xmax = int(math.floor(bbox[0] * w)), int(math.floor(bbox[2] * w))
    ymin, ymax = int(math.floor(bbox[1] * h)), int(math.floor(bbox[3] * h))
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or xmin >= xmax or ymin >= ymax:
        return False
    if max(bbox) > 1.0 or min(bbox) < 0.0:
        return False
    if max(h, w) / min(h, w) > max_ratio or max(xmax - xmin, ymax - ymin) / min(xmax - xmin, ymax - ymin) > max_ratio:
        return False
    return True


def load_lvr(metadata_file, num_proc=32):
    """Load LVR SFT data from its metadata file (``meta_data_lvr_sft_stage1.json``)."""
    metadata = pd.read_json(metadata_file)
    image_folder = metadata["image_folder"][0]
    data = load_dataset("json", data_files=list(metadata["data_path"]))["train"].shuffle(seed=42)
    data = data.map(lambda x: {
        "image": f"{image_folder}/{x['image'][0]}",
        "image_size": Image.open(f"{image_folder}/{x['image'][0]}").size,
        "bboxes": x["bboxes"][0],
    }, num_proc=num_proc)
    data = data.filter(lambda x: min(x["image_size"]) > 0)
    data = data.filter(lambda x: _valid_bbox(x["bboxes"], x["image_size"]))
    return data


def load_thinkmorph(root="ThinkMorph"):
    data = concatenate_datasets([load_dataset(f"{root}/{subset}", split="train") for subset in THINKMORPH_SUBSETS])
    return data.shuffle(seed=42)


def load_pixelreasoner(name="TIGER-Lab/PixelReasoner-SFT-Data"):
    data = load_dataset(name, split="train")
    # Video examples are not used.
    return data.filter(lambda x: not any(c["video"] is not None for c in x["message_list"][1]["content"]))


def load_train_dataset(dataset, train_file):
    if dataset == "lvr":
        return load_lvr(train_file)
    if dataset == "thinkmorph":
        return load_thinkmorph(train_file)
    if dataset == "pixelreasoner":
        return load_pixelreasoner(train_file)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def _mask_labels(messages, input_ids, attention_mask, tokenizer, last_turn_only):
    """Supervise only assistant tokens (and their ``<|im_end|>\\n``)."""
    labels = [-100] * len(input_ids)
    decoded_input = [tokenizer.decode(t) for t in input_ids]
    turns = messages[-1:] if last_turn_only else messages
    for msg in turns:
        if msg["role"] != "assistant":
            continue
        assert len(msg["content"]) == 1 and "text" in msg["content"][0]
        target = tokenizer.encode(f"{msg['content'][0]['text']}<|im_end|>\n", add_special_tokens=False)
        decoded_target = [tokenizer.decode(t) for t in target]
        for i in range(len(input_ids) - len(target) + 1):
            if attention_mask[i] == 1 and decoded_input[i:i + len(target)] == decoded_target:
                for j in range(i, min(i + len(target), len(input_ids))):
                    if attention_mask[j] == 1:
                        labels[j] = input_ids[j]
                break
    return labels


class PEARLCollator:
    """Builds the full / input / trajectory views for a batch.

    Label masking follows the paper runs: on LVR only the final answer turn is
    supervised, on ThinkMorph and PixelReasoner every assistant turn is.
    """

    def __init__(self, processor, dataset, predictors=4, image_root=None):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.predictors = predictors
        self.last_turn_only = dataset == "lvr"
        self.check_labels = dataset != "pixelreasoner"
        if dataset == "lvr":
            self.build_messages = get_messages_lvr
        elif dataset == "thinkmorph":
            self.build_messages = get_messages_thinkmorph
        elif dataset == "pixelreasoner":
            assert image_root, "--image_root is required for PixelReasoner"
            self.build_messages = lambda ex: get_messages_pixelreasoner(ex, image_root)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _apply_template(self, messages):
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def __call__(self, examples):
        processor = self.processor
        all_messages = [self.build_messages(ex) for ex in examples]

        # Full conversation (VLM loss).
        text = [self._apply_template(m) for m in all_messages]
        images = [process_vision_info(m)[0] for m in all_messages]
        batch = processor(text=text, images=images, return_tensors="pt", padding=True)
        max_length = batch["input_ids"].shape[1]
        labels = []
        for messages, ids, mask in zip(all_messages, list(batch["input_ids"]), list(batch["attention_mask"])):
            example_labels = _mask_labels(messages, ids, mask, self.tokenizer, self.last_turn_only)
            if self.check_labels:
                assert sum(example_labels) > -100 * len(example_labels), "No supervised tokens in example"
            labels.append(example_labels)
        batch["labels"] = torch.tensor(labels)

        # Input and trajectory views (JEPA loss), padded to the full-view length.
        user_messages = [get_user_messages(m, predictors=self.predictors) for m in all_messages]
        traj_messages = [get_trajectory_messages(m) for m in all_messages]
        traj_images = [process_vision_info(m)[0] for m in traj_messages]
        if not any(traj_images):
            traj_images = None
        batch_user = processor(text=[self._apply_template(m) for m in user_messages],
                               images=[process_vision_info(m)[0] for m in user_messages],
                               return_tensors="pt", padding="max_length", max_length=max_length)
        batch_traj = processor(text=[self._apply_template(m) for m in traj_messages],
                               images=traj_images,
                               return_tensors="pt", padding="max_length", max_length=max_length)
        batch_user["labels"] = torch.full_like(batch_user["input_ids"], -100)
        batch_traj["labels"] = torch.full_like(batch_traj["input_ids"], -100)

        batch = dict(batch)
        batch.update({f"{k}_user": v for k, v in batch_user.items()})
        batch.update({f"{k}_assistant": v for k, v in batch_traj.items()})
        return batch
