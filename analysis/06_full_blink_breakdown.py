"""
Full BLINK per-category breakdown with prediction distributions and
qualitative failure examples for all five categories.

Run with: python analysis/06_full_blink_breakdown.py
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
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

cats = ['Counting', 'IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
blink_ds = {}
for cat in cats:
    for d in load_dataset("BLINK-Benchmark/BLINK", cat)['val']:
        blink_ds[d['idx']] = d

print("=" * 80)
print("BLINK FULL PER-CATEGORY BREAKDOWN")
print("=" * 80)

for cat in cats:
    ids    = [(id_, e) for id_, e in our.items() if e.get('category') == cat]
    oc     = sum(acc_our(e['prediction'][0], e['label']) for _, e in ids)
    lc     = sum(acc_lvr(lvr[id_]['prediction'][0], lvr[id_]['label']) for id_, _ in ids if id_ in lvr)
    pc     = sum(extract_pr(pr[id_]['prediction']) == pr[id_]['label'] for id_, _ in ids if id_ in pr)
    n      = len(ids)
    our_pd = Counter(pred_our(e['prediction'][0]) for _, e in ids)
    lvr_pd = Counter(pred_our(lvr[id_]['prediction'][0]) for id_, _ in ids if id_ in lvr)
    pr_pd  = Counter(extract_pr(pr[id_]['prediction']) for id_, _ in ids if id_ in pr)
    true_d = Counter(e['label'] for _, e in ids)

    print(f"\n{'─'*80}")
    print(f"{cat}  (n={n})  PEARL={oc/n:.1%}  LVR={lc/n:.1%}  PR={pc/n:.1%}")
    print(f"  True:      {dict(sorted(true_d.items()))}")
    print(f"  PEARL: {dict(sorted(our_pd.items()))}")
    print(f"  LVR-7B:    {dict(sorted(lvr_pd.items()))}")
    print(f"  PR-nocot:  {dict(sorted(pr_pd.items()))}")

    # Failures where another model wins
    worse = []
    for id_, e in ids:
        if acc_our(e['prediction'][0], e['label']): continue
        lc_i  = acc_lvr(lvr[id_]['prediction'][0], e['label']) if id_ in lvr else False
        prc_i = extract_pr(pr[id_]['prediction']) == e['label'] if id_ in pr else False
        if lc_i or prc_i:
            lvr_p = pred_our(lvr[id_]['prediction'][0]) if id_ in lvr else '?'
            pr_p  = extract_pr(pr[id_]['prediction']) if id_ in pr else '?'
            num   = int(id_.split('_')[-1])
            dat   = blink_ds.get(id_, blink_ds.get(num, {}))
            worse.append({
                'id': id_, 'label': e['label'],
                'our': pred_our(e['prediction'][0]), 'lvr': lvr_p, 'pr': pr_p,
                'lc': lc_i, 'prc': prc_i,
                'q': dat.get('question', ''),
                'opts': dat.get('choices', []),
                'pr_text': pr[id_]['prediction'][:350] if id_ in pr else '',
            })

    if worse:
        print(f"\n  Failures where LVR or PR correct ({len(worse)} cases):")
        for f in worse[:5]:
            better = (' LVR✓' if f['lc'] else '') + (' PR✓' if f['prc'] else '')
            print(f"    {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}' [{better}]")
            print(f"    Q: {f['q'][:100]}")
            if f['prc'] and cat in ('IQ_Test', 'Relative_Reflectance'):
                print(f"    PR: {f['pr_text'].replace(chr(10), ' ')[:300]}")
