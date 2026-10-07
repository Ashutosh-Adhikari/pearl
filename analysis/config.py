"""Shared result paths and answer-extraction helpers for the analysis scripts.

Result paths are read from a JSON file: ``$PEARL_ANALYSIS_PATHS`` if set,
otherwise ``analysis/paths.json`` (copy ``paths.example.json`` to start).

* ``PEARL_*`` / ``LVR_*``: prediction files written by ``eval/evaluate.py``
  (a JSON list of ``{"id", "prediction", "label", "category"}``).
* ``PR_*``: PixelReasoner predictions (``{"id", "prediction", ...}`` with its
  long-form response) and ``PR_*_TRACES`` with the full reasoning traces.
* ``MMVP_CSV``: ``Questions.csv`` from the MMVP benchmark.
"""

import json
import os
import re
from pathlib import Path

_PATHS_FILE = os.environ.get("PEARL_ANALYSIS_PATHS", str(Path(__file__).parent / "paths.json"))
with open(_PATHS_FILE) as f:
    _paths = json.load(f)

PEARL_BLINK = _paths["PEARL_BLINK"]
PEARL_VSTAR = _paths["PEARL_VSTAR"]
PEARL_MMVP = _paths["PEARL_MMVP"]

LVR_BLINK = _paths["LVR_BLINK"]
LVR_VSTAR = _paths["LVR_VSTAR"]
LVR_MMVP = _paths["LVR_MMVP"]

PR_BLINK = _paths["PR_BLINK"]
PR_BLINK_TRACES = _paths["PR_BLINK_TRACES"]
PR_VSTAR = _paths["PR_VSTAR"]
PR_VSTAR_TRACES = _paths["PR_VSTAR_TRACES"]
PR_MMVP = _paths["PR_MMVP"]
PR_MMVP_TRACES = _paths["PR_MMVP_TRACES"]

MMVP_CSV = _paths["MMVP_CSV"]


# ── Accuracy / prediction extraction ─────────────────────────────────────

def acc_our(response: str, ground_truth: str) -> bool:
    """Accuracy function used by eval/evaluate.py."""
    given = response.split('<answer>')[-1].split('</answer')[0].strip()
    if ground_truth in given:
        return True
    if ' ' in given:
        given = given.split(' ')[0]
    if len(given) > 1:
        given = given[0]
    return given == ground_truth


def pred_our(response: str) -> str:
    """Extract single-letter prediction from our model's output."""
    given = response.split('<answer>')[-1].split('</answer')[0].strip()
    if ' ' in given:
        given = given.split(' ')[0]
    if len(given) > 1:
        given = given[0]
    return given


def acc_lvr(response: str, ground_truth: str) -> bool:
    """Accuracy function of the original LVR evaluation (exact first letter)."""
    given = response.split('<answer>')[-1].split('</answer')[0].strip()
    if ' ' in given:
        given = given.split(' ')[0]
    if len(given) > 1:
        given = given[0]
    return given == ground_truth


def extract_pr(text: str) -> str:
    """Extract answer letter from PixelReasoner's long-form output."""
    m = re.search(r'\\boxed\{([A-D])\}', text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r'(?:answer is|answer:)\s*[\(]?([A-D])[\)]?', text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([A-D])\b[\.\s]*$', text.strip(), re.I)
    if m:
        return m.group(1).upper()
    letters = re.findall(r'\b([A-D])\b', text.upper())
    return letters[-1] if letters else '?'
