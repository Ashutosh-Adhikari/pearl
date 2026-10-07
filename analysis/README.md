# Analysis scripts

Per-category and qualitative comparison of PEARL with LVR and PixelReasoner on BLINK, V* and MMVP.

1. Produce prediction files for PEARL and LVR with `eval/evaluate.py`, and PixelReasoner predictions
   and traces with the [PixelReasoner](https://github.com/TIGER-AI-Lab/Pixel-Reasoner) evaluation code
   (one JSON list of `{"id", "prediction", "label"}` per benchmark, plus a traces file of
   `{"id", "turns": [...]}`).
2. `cp analysis/paths.example.json analysis/paths.json` and fill in the paths
   (or point `PEARL_ANALYSIS_PATHS` at another file).
3. Run any script from the repository root, e.g. `python analysis/01_overall_scores.py`.

| Script | Output |
|---|---|
| `01_overall_scores.py` | overall and per-category accuracy, prediction distributions |
| `02_relative_reflectance_failures.py` | BLINK Relative Reflectance failure breakdown |
| `03_iq_test_failures.py` | BLINK IQ Test failure breakdown |
| `04_vstar_failures.py` | V* failures where another model is correct |
| `05_mmvp_analysis.py` | MMVP accuracy, calibration and question-type breakdown |
| `06_full_blink_breakdown.py` | all BLINK categories with examples |
| `07_wins_analysis.py` | cases where only PEARL is correct |
