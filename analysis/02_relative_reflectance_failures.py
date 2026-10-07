"""
Relative Reflectance failure analysis: PEARL vs LVR vs PR.

Prints:
  - Breakdown of our failures (unique vs shared)
  - Cases where we fail but others succeed, with PR reasoning
  - Prediction distribution tables

Run with: python analysis/02_relative_reflectance_failures.py
"""
import json
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PEARL_BLINK, LVR_BLINK, PR_BLINK,
    acc_our, acc_lvr, extract_pr, pred_our,
)
from datasets import load_dataset


def load(path):
    with open(path) as f:
        return {d['id']: d for d in json.load(f)}


our = load(PEARL_BLINK)
lvr = load(LVR_BLINK)
pr  = load(PR_BLINK)

blink_rr = {d['idx']: d for d in load_dataset("BLINK-Benchmark/BLINK", "Relative_Reflectance")['val']}

rr_ids = [(id_, e) for id_, e in our.items() if e.get('category') == 'Relative_Reflectance']

fails = []
for id_, e in rr_ids:
    if acc_our(e['prediction'][0], e['label']):
        continue
    lvr_p = pred_our(lvr[id_]['prediction'][0]) if id_ in lvr else '?'
    pr_p  = extract_pr(pr[id_]['prediction'])   if id_ in pr  else '?'
    lc    = acc_lvr(lvr[id_]['prediction'][0], e['label']) if id_ in lvr else False
    prc   = pr_p == e['label']
    num   = int(id_.split('_')[-1])
    dat   = blink_rr.get(id_, blink_rr.get(num, {}))
    fails.append({
        'id': id_, 'label': e['label'],
        'our': pred_our(e['prediction'][0]), 'lvr': lvr_p, 'pr': pr_p,
        'lc': lc, 'prc': prc,
        'q': dat.get('question', ''),
        'opts': dat.get('choices', []),
        'pr_text': pr[id_]['prediction'] if id_ in pr else '',
    })

we_worst   = [f for f in fails if f['lc'] and f['prc']]
pr_better  = [f for f in fails if f['prc'] and not f['lc']]
lvr_better = [f for f in fails if f['lc'] and not f['prc']]
all_wrong  = [f for f in fails if not f['lc'] and not f['prc']]

print("=" * 80)
print("RELATIVE REFLECTANCE — our failures breakdown")
print("=" * 80)
print(f"Total failures: {len(fails)} / {len(rr_ids)}  (acc={1-len(fails)/len(rr_ids):.1%})")
print(f"  We wrong, both others right: {len(we_worst)}")
print(f"  We wrong, only PR right:     {len(pr_better)}")
print(f"  We wrong, only LVR right:    {len(lvr_better)}")
print(f"  All three wrong:             {len(all_wrong)}")

# Prediction distributions
our_pd = Counter(pred_our(e['prediction'][0]) for _, e in rr_ids)
lvr_pd = Counter(pred_our(lvr[id_]['prediction'][0]) for id_, _ in rr_ids if id_ in lvr)
pr_pd  = Counter(extract_pr(pr[id_]['prediction']) for id_, _ in rr_ids if id_ in pr)
true_d = Counter(e['label'] for _, e in rr_ids)
print(f"\nPrediction distributions (true: {dict(sorted(true_d.items()))}):")
print(f"  PEARL: {dict(sorted(our_pd.items()))}")
print(f"  LVR-7B:    {dict(sorted(lvr_pd.items()))}")
print(f"  PR-nocot:  {dict(sorted(pr_pd.items()))}")

print(f"\n{'─'*80}")
print("UNIQUE FAILURES — we wrong, both LVR and PR right")
print(f"{'─'*80}")
for f in we_worst:
    print(f"\n  {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'")
    print(f"  Q:    {f['q']}")
    print(f"  Opts: {f['opts']}")
    print(f"  PR:   {f['pr_text'][:350].replace(chr(10), ' ')}")

print(f"\n{'─'*80}")
print("PR BEATS US — we+LVR over-hedge to C, PR commits to right answer")
print(f"{'─'*80}")
c_both_c = [f for f in pr_better if f['our'] == 'C' and f['lvr'] == 'C']
c_we_a   = [f for f in pr_better if f['our'] == 'A']
other_pr = [f for f in pr_better if f not in c_both_c and f not in c_we_a]
print(f"\n  [We+LVR say C, PR says {f['pr']} correctly] — {len(c_both_c)} cases")
for f in c_both_c[:6]:
    print(f"\n  {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'")
    print(f"  Q:    {f['q']}")
    print(f"  Opts: {f['opts']}")
    print(f"  PR:   {f['pr_text'][:400].replace(chr(10), ' ')}")
print(f"\n  [We say A, PR says correct] — {len(c_we_a)} cases")
for f in c_we_a[:4]:
    print(f"\n  {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'")
    print(f"  Q:    {f['q']}")
    print(f"  PR:   {f['pr_text'][:350].replace(chr(10), ' ')}")

print(f"\n{'─'*80}")
print("LVR BEATS US — we over-commit, LVR hedges correctly (or vice versa)")
print(f"{'─'*80}")
for f in lvr_better[:6]:
    print(f"\n  {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'")
    print(f"  Q:    {f['q']}")
    print(f"  Opts: {f['opts']}")
