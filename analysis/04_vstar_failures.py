"""
VSTAR failure analysis: PEARL vs LVR vs PR.

Prints failures in direct_attributes and relative_position where another
model is correct, with full question text.

Run with: python analysis/04_vstar_failures.py
"""
import json
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PEARL_VSTAR, LVR_VSTAR, PR_VSTAR, PR_VSTAR_TRACES,
    acc_our, acc_lvr, extract_pr, pred_our,
)
from datasets import load_dataset


def load(path):
    with open(path) as f:
        return {d['id']: d for d in json.load(f)}


our = load(PEARL_VSTAR)
lvr = load(LVR_VSTAR)
pr  = load(PR_VSTAR)
pr_traces = load(PR_VSTAR_TRACES)
vstar_ds  = {d['question_id']: d for d in load_dataset("craigwu/vstar_bench")['test']}

fails = []
for id_, e in our.items():
    if acc_our(e['prediction'][0], e['label']):
        continue
    lvr_p = pred_our(lvr[id_]['prediction'][0]) if id_ in lvr else '?'
    pr_p  = extract_pr(pr[id_]['prediction'])   if id_ in pr  else '?'
    lc    = acc_lvr(lvr[id_]['prediction'][0], e['label']) if id_ in lvr else False
    prc   = pr_p == e['label']
    dat   = vstar_ds.get(id_, {})
    # PR traces contain the raw assistant turn for reasoning
    pr_turns = pr_traces[id_]['turns'] if id_ in pr_traces else []
    pr_reasoning = pr_turns[0]['text'] if pr_turns else (pr[id_]['prediction'] if id_ in pr else '')
    fails.append({
        'id': id_, 'cat': e.get('category', '?'), 'label': e['label'],
        'our': pred_our(e['prediction'][0]), 'lvr': lvr_p, 'pr': pr_p,
        'lc': lc, 'prc': prc,
        'q': dat.get('text', ''),
        'pr_reasoning': pr_reasoning,
    })

vs_fails = [f for f in fails if f['lc'] or f['prc']]

print("=" * 80)
print("VSTAR FAILURES — where LVR or PR is correct")
print("=" * 80)

for cat in ['direct_attributes', 'relative_position']:
    cat_fails = [f for f in vs_fails if f['cat'] == cat]
    lvr_only  = [f for f in cat_fails if f['lc'] and not f['prc']]
    pr_only   = [f for f in cat_fails if f['prc'] and not f['lc']]
    both      = [f for f in cat_fails if f['lc'] and f['prc']]

    print(f"\n{'─'*80}")
    print(f"{cat.upper()} ({len(cat_fails)} failures where another wins)")
    print(f"  LVR only right: {len(lvr_only)}, PR only right: {len(pr_only)}, Both right: {len(both)}")
    print(f"{'─'*80}")
    for f in cat_fails:
        better = []
        if f['lc']:  better.append('LVR✓')
        if f['prc']: better.append('PR✓')
        print(f"\n  id={f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'  [{', '.join(better)}]")
        print(f"  Q: {f['q'][:150]}")
        if f['prc']:
            print(f"  PR reasoning: {f['pr_reasoning'][:300].replace(chr(10), ' ')}")

# Summary of prediction bias for relative_position
rp_all = [(id_, e) for id_, e in our.items() if e.get('category') == 'relative_position']
rp_pd  = Counter(pred_our(e['prediction'][0]) for _, e in rp_all)
rp_true= Counter(e['label'] for _, e in rp_all)
print(f"\n{'─'*80}")
print("RELATIVE POSITION — prediction distribution analysis")
print(f"  True labels: {dict(sorted(rp_true.items()))}")
print(f"  Our preds:   {dict(sorted(rp_pd.items()))}")
# Cases where answer=A and we say B
a_label_b_pred = [(id_, e) for id_, e in rp_all if e['label']=='A' and pred_our(e['prediction'][0])=='B']
b_label_a_pred = [(id_, e) for id_, e in rp_all if e['label']=='B' and pred_our(e['prediction'][0])=='A']
print(f"  label=A, we say B: {len(a_label_b_pred)} (B-bias misses)")
print(f"  label=B, we say A: {len(b_label_a_pred)} (A-bias misses)")
