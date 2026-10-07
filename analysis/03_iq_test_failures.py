"""
IQ_Test failure analysis: PEARL vs LVR vs PR.

Prints:
  - B-collapse statistics
  - Cases where LVR reaches correct C/D that we and PR miss
  - Cases where PR reasoning finds the right answer

Run with: python analysis/03_iq_test_failures.py
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

blink_iq = {d['idx']: d for d in load_dataset("BLINK-Benchmark/BLINK", "IQ_Test")['val']}

iq_ids = [(id_, e) for id_, e in our.items() if e.get('category') == 'IQ_Test']

our_pd = Counter(pred_our(e['prediction'][0]) for _, e in iq_ids)
lvr_pd = Counter(pred_our(lvr[id_]['prediction'][0]) for id_, _ in iq_ids if id_ in lvr)
pr_pd  = Counter(extract_pr(pr[id_]['prediction']) for id_, _ in iq_ids if id_ in pr)
true_d = Counter(e['label'] for _, e in iq_ids)

print("=" * 80)
print("IQ_TEST FAILURE ANALYSIS")
print("=" * 80)
oc = sum(acc_our(e['prediction'][0], e['label']) for _, e in iq_ids)
lc = sum(acc_lvr(lvr[id_]['prediction'][0], lvr[id_]['label']) for id_, _ in iq_ids if id_ in lvr)
pc = sum(extract_pr(pr[id_]['prediction']) == pr[id_]['label'] for id_, _ in iq_ids if id_ in pr)
n  = len(iq_ids)
print(f"  PEARL: {oc}/{n} = {oc/n:.1%}")
print(f"  LVR-7B:    {lc}/{n} = {lc/n:.1%}")
print(f"  PR-nocot:  {pc}/{n} = {pc/n:.1%}")

print(f"\nPrediction distributions (true: {dict(sorted(true_d.items()))}):")
print(f"  PEARL: {dict(sorted(our_pd.items()))}")
print(f"  LVR-7B:    {dict(sorted(lvr_pd.items()))}")
print(f"  PR-nocot:  {dict(sorted(pr_pd.items()))}")
print(f"\n  B-collapse: PEARL predicts B {our_pd['B']}/{n} = {our_pd['B']/n:.1%} of the time")
print(f"              PR-nocot  predicts B {pr_pd['B']}/{n}  = {pr_pd['B']/n:.1%} of the time")

fails = []
for id_, e in iq_ids:
    oc_i  = acc_our(e['prediction'][0], e['label'])
    lc_i  = acc_lvr(lvr[id_]['prediction'][0], lvr[id_]['label']) if id_ in lvr else False
    prc_i = extract_pr(pr[id_]['prediction']) == e['label'] if id_ in pr else False
    if oc_i:
        continue
    lvr_p = pred_our(lvr[id_]['prediction'][0]) if id_ in lvr else '?'
    pr_p  = extract_pr(pr[id_]['prediction'])   if id_ in pr  else '?'
    num   = int(id_.split('_')[-1])
    dat   = blink_iq.get(id_, blink_iq.get(num, {}))
    fails.append({
        'id': id_, 'label': e['label'],
        'our': pred_our(e['prediction'][0]), 'lvr': lvr_p, 'pr': pr_p,
        'lc': lc_i, 'prc': prc_i,
        'q': dat.get('question', ''),
        'opts': dat.get('choices', []),
        'pr_text': pr[id_]['prediction'] if id_ in pr else '',
    })

lvr_wins = [f for f in fails if f['lc'] and not f['prc']]
pr_wins  = [f for f in fails if f['prc'] and not f['lc']]
both_win = [f for f in fails if f['lc'] and f['prc']]
none_win = [f for f in fails if not f['lc'] and not f['prc']]

print(f"\nFailure breakdown ({len(fails)} failures):")
print(f"  LVR only right: {len(lvr_wins)}, PR only right: {len(pr_wins)}, Both right: {len(both_win)}, All wrong: {len(none_win)}")

print(f"\n{'─'*80}")
print("LVR BEATS US — latent steps reach C/D, we and PR stay at B")
print(f"{'─'*80}")
for f in lvr_wins[:8]:
    print(f"\n  {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'")
    print(f"  Q:    {f['q']}")
    print(f"  Opts: {f['opts']}")

print(f"\n{'─'*80}")
print("PR BEATS US — explicit reasoning chains find the pattern")
print(f"{'─'*80}")
for f in pr_wins[:6]:
    print(f"\n  {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'")
    print(f"  Q:    {f['q']}")
    print(f"  Opts: {f['opts']}")
    print(f"  PR reasoning: {f['pr_text'][:450].replace(chr(10), ' ')}")

print(f"\n{'─'*80}")
print("BOTH LVR AND PR RIGHT — our model is uniquely wrong")
print(f"{'─'*80}")
for f in both_win:
    print(f"\n  {f['id']}  truth={f['label']}  our='{f['our']}'  lvr='{f['lvr']}'  pr='{f['pr']}'")
    print(f"  Q:    {f['q']}")
    print(f"  Opts: {f['opts']}")
