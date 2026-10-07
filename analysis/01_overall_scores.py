"""
Compute overall 3-way accuracy table for all benchmarks.
Run with: python analysis/01_overall_scores.py
"""
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PEARL_BLINK, PEARL_VSTAR, PEARL_MMVP,
    LVR_BLINK, LVR_VSTAR, LVR_MMVP,
    PR_BLINK, PR_VSTAR, PR_MMVP,
    acc_our, acc_lvr, extract_pr, pred_our,
)


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {d['id']: d for d in data}


def score_bench(our, lvr, pr, acc_fn_our, acc_fn_lvr, key='prediction'):
    common = set(our) & set(lvr) & set(pr)
    oc = sum(acc_fn_our(our[i][key][0] if isinstance(our[i][key], list) else our[i][key], our[i]['label'])
             for i in common)
    lc = sum(acc_fn_lvr(lvr[i][key][0] if isinstance(lvr[i][key], list) else lvr[i][key], lvr[i]['label'])
             for i in common)
    pc = sum(extract_pr(pr[i][key]) == pr[i]['label'] for i in common)
    n  = len(common)
    return oc, lc, pc, n


our_b = load(PEARL_BLINK); lvr_b = load(LVR_BLINK); pr_b = load(PR_BLINK)
our_v = load(PEARL_VSTAR); lvr_v = load(LVR_VSTAR); pr_v = load(PR_VSTAR)
our_m = load(PEARL_MMVP);  lvr_m = load(LVR_MMVP);  pr_m = load(PR_MMVP)

ob, lb, pb, nb = score_bench(our_b, lvr_b, pr_b, acc_our, acc_lvr, key='prediction')
ov, lv, pv, nv = score_bench(our_v, lvr_v, pr_v, acc_our, acc_lvr, key='prediction')
om, lm, pm, nm = score_bench(our_m, lvr_m, pr_m, acc_our, acc_lvr, key='prediction')

print("=" * 70)
print(f"{'Benchmark':<12} {'PEARL':>12} {'LVR-7B':>10} {'PR-nocot':>10} {'n':>6}")
print("-" * 70)
print(f"{'BLINK':<12} {ob/nb*100:>11.1f}% {lb/nb*100:>9.1f}% {pb/nb*100:>9.1f}% {nb:>6}")
print(f"{'VSTAR':<12} {ov/nv*100:>11.1f}% {lv/nv*100:>9.1f}% {pv/nv*100:>9.1f}% {nv:>6}")
print(f"{'MMVP':<12} {om/nm*100:>11.1f}% {lm/nm*100:>9.1f}% {pm/nm*100:>9.1f}% {nm:>6}")
print("=" * 70)

# BLINK per-category
print("\nBLINK per-category:")
cats = ['Counting', 'IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
print(f"  {'Category':<25} {'PEARL':>10} {'LVR-7B':>8} {'PR-nocot':>8} {'n':>5}")
for cat in cats:
    ids = [i for i, d in our_b.items() if d.get('category') == cat]
    oc = sum(acc_our(our_b[i]['prediction'][0], our_b[i]['label']) for i in ids)
    lc = sum(acc_lvr(lvr_b[i]['prediction'][0], lvr_b[i]['label']) for i in ids if i in lvr_b)
    pc = sum(extract_pr(pr_b[i]['prediction']) == pr_b[i]['label'] for i in ids if i in pr_b)
    n  = len(ids)
    print(f"  {cat:<25} {oc/n*100:>9.1f}% {lc/n*100:>7.1f}% {pc/n*100:>7.1f}% {n:>5}")

# VSTAR per-category
print("\nVSTAR per-category:")
for cat in ['direct_attributes', 'relative_position']:
    ids = [i for i, d in our_v.items() if d.get('category') == cat]
    oc = sum(acc_our(our_v[i]['prediction'][0], our_v[i]['label']) for i in ids)
    lc = sum(acc_lvr(lvr_v[i]['prediction'][0], lvr_v[i]['label']) for i in ids if i in lvr_v)
    pc = sum(extract_pr(pr_v[i]['prediction']) == pr_v[i]['label'] for i in ids if i in pr_v)
    n  = len(ids)
    print(f"  {cat:<25} {oc/n*100:>9.1f}% {lc/n*100:>7.1f}% {pc/n*100:>7.1f}% {n:>5}")

# Prediction distributions
print("\nPrediction distributions — MMVP (true: A=150 B=150):")
for name, data, fn in [("PEARL", our_m, pred_our), ("LVR-7B", lvr_m, pred_our),
                        ("PR-nocot", pr_m, extract_pr)]:
    preds = Counter(fn(d['prediction'][0] if isinstance(d['prediction'], list) else d['prediction'])
                    for d in data.values())
    print(f"  {name:<12}: {dict(sorted(preds.items()))}")

print("\nPrediction distributions — BLINK IQ_Test (true: A=30 B=40 C=40 D=40):")
for name, data, fn, key in [("PEARL", our_b, pred_our, 'prediction'),
                              ("PR-nocot", pr_b, extract_pr, 'prediction')]:
    ids = [i for i, d in our_b.items() if d.get('category') == 'IQ_Test']
    preds = Counter(fn(data[i][key][0] if isinstance(data[i][key], list) else data[i][key])
                    for i in ids if i in data)
    print(f"  {name:<12}: {dict(sorted(preds.items()))}")
