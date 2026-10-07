# PEARL: Multimodal Latent Reasoning via Predictive Embeddings

Code for **PEARL** (**P**redictive **E**mbedding **A**lignment for **R**easoning in **L**atent space),
from [*Multimodal Latent Reasoning via Predictive Embeddings*](https://openreview.net/pdf?id=E44oGC9FRV)
by Ashutosh Adhikari and Mirella Lapata.

PEARL fine-tunes a vision-language model on expert tool-use trajectories (crops, bounding boxes,
highlights, ...) without ever calling a tool at inference. Instead of reconstructing the tool outputs,
the model learns to *predict the embedding of the trajectory* from the image-question pair:

```
L_PEARL = L_VLM + λ · (L_JEPA + L_NextLat)
```

* **L_VLM** — next-token prediction on the full conversation (preserves generation).
* **L_JEPA** — the image-question pair followed by K learnable `[PRED]` tokens is encoded in one
  forward pass, the trajectory (image + tool calls/outputs, no question) in another. The last-token
  hidden state of the first is aligned to the stop-gradient last-token hidden state of the second.
  The predictor shares the VLM's weights.
* **L_NextLat** — a small dynamics head predicts the next hidden state from the current hidden state
  and the next token embedding; the prediction is matched to the true next state (SmoothL1) and decoded
  through a frozen copy of the LM head (cross-entropy).

At inference PEARL is a plain Qwen-VL model: no predictor tokens, no extra heads, no tool calls.

## Installation

```bash
git clone https://github.com/Ashutosh-Adhikari/pearl.git && cd pearl
conda create -n pearl python=3.11 -y && conda activate pearl
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

`requirements.txt` pins the versions used for the Qwen2.5-VL experiments. Qwen3-VL needs
`transformers>=4.57` (`pip install -U "transformers>=4.57"`).

## Data

| Regime | Dataset | `--dataset` | `--train_file` |
|---|---|---|---|
| single tool type, single call | [LVR](https://github.com/VincentLeebang/lvr) SFT data | `lvr` | path to `meta_data_lvr_sft_stage1.json` (LVR metadata file listing `data_path` and `image_folder`) |
| multiple tool types, single call | [ThinkMorph](https://huggingface.co/ThinkMorph) (`Visual_Search`, `Jigsaw_Assembly`, `Chart_Refocus`, `Spatial_Navigation`) | `thinkmorph` | `ThinkMorph` (Hub org) or a local root holding the four subsets |
| single tool type, multiple calls | [PixelReasoner SFT](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-SFT-Data) | `pixelreasoner` | `TIGER-Lab/PixelReasoner-SFT-Data`; also pass `--image_root` pointing to a local snapshot that contains its `images/` folder |

Follow the LVR repository for downloading and preparing its SFT data and images.

Evaluation uses [V\*](https://huggingface.co/datasets/craigwu/vstar_bench) (download the dataset repo
locally for the images), [MMVP](https://huggingface.co/datasets/MMVP/MMVP) (`Questions.csv` and
`MMVP_Images/`) and [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) (loaded from the Hub).

## Training

The scripts in `scripts/` use the settings of the paper runs (LoRA r=64, α=128, λ=0.2, K=4, lr 1e-5,
cosine schedule, DeepSpeed ZeRO-2 on 4 GPUs):

```bash
LVR_META=/path/to/meta_data_lvr_sft_stage1.json bash scripts/train_lvr.sh
bash scripts/train_thinkmorph.sh
PR_IMAGES=/path/to/PixelReasoner-SFT-Data bash scripts/train_pixelreasoner.sh
```

Override the model with `MODEL=Qwen/Qwen3-VL-4B-Instruct`, the number of GPUs with `NGPUS`, and pass
any extra `pearl.train` flag after the script. Run `python -m pearl.train --help` for all options.

Ablations from the paper:

| Flag | Effect |
|---|---|
| `--no_nextlat` | drop L_NextLat |
| `--no_jepa` | drop L_JEPA |
| `--predictors K` | number of `[PRED]` tokens |
| `--jepa_loss {cosine,smoothl1,l2,mse}` | distance for L_JEPA |

> **Note on the JEPA distance.** The results in the paper were obtained with the **cosine** distance
> (`1 − cos`) for L_JEPA, which is the default here. The paper text describes SmoothL1; SmoothL1 is the
> distance used in L_NextLat. `--jepa_loss smoothl1` is provided for completeness but was not used for
> the reported numbers.

Checkpoints are LoRA adapters (`<output_dir>/checkpoint-N`). At the end of training the adapter is
merged into the base model and saved to `<output_dir>` (disable with `--no_merge`); any intermediate
checkpoint can be merged with

```bash
python eval/merge_lora.py --checkpoint runs/<run>/checkpoint-N --output_dir merged/<run>
```

## Evaluation

```bash
VSTAR_DIR=data/vstar_bench MMVP_DIR=data/MMVP bash scripts/eval.sh runs/<run>/checkpoint-N [more checkpoints...]
```

Merged models, LoRA checkpoints and Hub ids are all accepted. Answers are generated greedily (4 new
tokens) and scored on the predicted option letter. Predictions are written to
`results/<run>/<benchmark>.json` and a `summary.json` with overall and per-category accuracy.

`--preset` selects the prompt format:

| Preset | Task instruction | Generation prefix |
|---|---|---|
| `default` | BLINK only: "Answer with the option's letter from the given choices directly." | `<answer>` |
| `instruct` | the same instruction on all benchmarks | `<answer>` |
| `instruct_no_prefix` | MMVP and BLINK | none |
| `thinkmorph` | "... directly between `<answer></answer>` tags." on all benchmarks | `<answer>` |
| `boxed` | "Answer with the correct option's letter directly within `\boxed{}`." (adds BLINK Object_Localization) | none |

## Analysis

`analysis/` contains the scripts for the per-category and qualitative comparison of PEARL against LVR
and PixelReasoner; see [analysis/README.md](analysis/README.md).

## Repository layout

```
pearl/          training code (data + collator, model setup, PEARL trainer, entry point)
eval/           benchmark evaluation and LoRA merging
analysis/       qualitative / per-category analysis scripts
configs/        DeepSpeed ZeRO-2 config
scripts/        launch scripts with the paper settings
```

## Acknowledgements

This code builds directly on [LLM-JEPA](https://github.com/rbalestr-lab/llm-jepa) by Hai Huang,
Yann LeCun and Randall Balestriero: the JEPA trainer, predictor tokens and last-token embedding
originate there, and we extend them to vision-language models, tool-use trajectories and the NextLat
objective. The evaluation code is adapted from [LVR](https://github.com/VincentLeebang/lvr). We thank
the authors of LVR, ThinkMorph and PixelReasoner for releasing their data. Both upstream projects
are licensed under Apache-2.0; see [NOTICE](NOTICE).

## Citation

```bibtex
@inproceedings{
adhikari2026multimodal,
title={Multimodal Latent Reasoning via Predictive  Embeddings},
author={Ashutosh Adhikari and Mirella Lapata},
booktitle={Third Conference on Language Modeling},
year={2026},
url={https://openreview.net/forum?id=E44oGC9FRV}
}
```

Please also cite LLM-JEPA, on which this code is based:

```bibtex
@article{huang2025llmjepa,
  title   = {{LLM-JEPA}: Large Language Models Meet Joint Embedding Predictive Architectures},
  author  = {Huang, Hai and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2509.14252},
  year    = {2025}
}
```

and, if you use the LVR data or evaluation:

```bibtex
@article{li2025lvr,
  title   = {Latent Visual Reasoning},
  author  = {Li, Bangzheng and Sun, Ximeng and Liu, Jiang and Wang, Ze and Wu, Jialian and Yu, Xiaodong and Chen, Hao and Barsoum, Emad and Chen, Muhao and Liu, Zicheng},
  journal = {arXiv preprint arXiv:2509.24251},
  year    = {2025}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
