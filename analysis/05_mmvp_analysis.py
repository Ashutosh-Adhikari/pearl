"""
MMVP analysis: PEARL vs LVR vs PR.

Prints:
  - Overall accuracy and calibration
  - Question-type breakdown (shadow, orientation, visibility, counting, other)
  - PR wins vs our wins with question text

Run with: python analysis/05_mmvp_analysis.py
"""
import json, csv
from pathlib import Path
from collections import Counter, defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PEARL_MMVP, LVR_MMVP, PR_MMVP, PR_MMVP_TRACES, MMVP_CSV,
    acc_our, acc_lvr, extract_pr, pred_our,
)


def load(path):
    with open(path) as f:
        return {d['id']: d for d in json.load(f)}


our = load(PEARL_MMVP)
lvr = load(LVR_MMVP)
pr  = load(PR_MMVP)

mmvp_q = {}
with open(MMVP_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        mmvp_q[int(row['Index'])] = row

# ── Overall scores ────────────────────────────────────────────────────────
n  = len(our)
oc = sum(acc_our(e['prediction'][0], e['label']) for e in our.values())
lc = sum(acc_lvr(lvr[i]['prediction'][0], lvr[i]['label']) for i in our if i in lvr)
pc = sum(extract_pr(pr[i]['prediction']) == pr[i]['label'] for i in our if i in pr)

our_pd = Counter(pred_our(e['prediction'][0]) for e in our.values())
lvr_pd = Counter(pred_our(lvr[i]['prediction'][0]) for i in our if i in lvr)
pr_pd  = Counter(extract_pr(pr[i]['prediction']) for i in our if i in pr)
true_d = Counter(e['label'] for e in our.values())

print("=" * 80)
print("MMVP ANALYSIS")
print("=" * 80)
print(f"  PEARL: {oc}/{n} = {oc/n:.1%}   preds: {dict(sorted(our_pd.items()))}")
print(f"  LVR-7B:    {lc}/{n} = {lc/n:.1%}   preds: {dict(sorted(lvr_pd.items()))}")
print(f"  PR-nocot:  {pc}/{n} = {pc/n:.1%}   preds: {dict(sorted(pr_pd.items()))}")
print(f"  True dist: {dict(sorted(true_d.items()))}")

# ── PR wins vs our wins ───────────────────────────────────────────────────
pr_wins, our_wins, both_wrong, both_right = [], [], [], []
for id_, e in our.items():
    if id_ not in pr: continue
    oc_i  = acc_our(e['prediction'][0], e['label'])
    prc_i = extract_pr(pr[id_]['prediction']) == e['label']
    q = mmvp_q.get(id_, {})
    entry = {
        'id': id_, 'label': e['label'],
        'our': pred_our(e['prediction'][0]),
        'pr':  extract_pr(pr[id_]['prediction']),
        'q':   q.get('Question', '?'),
        'opts': q.get('Options', '?'),
    }
    if prc_i and not oc_i:   pr_wins.append(entry)
    elif oc_i and not prc_i: our_wins.append(entry)
    elif not oc_i and not prc_i: both_wrong.append(entry)
    else: both_right.append(entry)

print(f"\n  PR wins (PR✓ our✗): {len(pr_wins)}   Our wins (our✓ PR✗): {len(our_wins)}")
print(f"  Both wrong: {len(both_wrong)}   Both right: {len(both_right)}")

# ── Question-type breakdown ───────────────────────────────────────────────
def categorize(q_text):
    q = q_text.lower()
    if any(k in q for k in ['shadow', 'reflect', 'light', 'bright', 'darker', 'lighter', 'illuminat']): return 'shadow/light'
    if any(k in q for k in ['left', 'right', 'facing', 'direction', 'toward', 'away']):                  return 'orientation'
    if any(k in q for k in ['visible', 'see', 'show', 'appear', 'entire', 'submerged']):                 return 'visibility'
    if any(k in q for k in ['how many', 'count', 'number', 'range']):                                    return 'counting'
    return 'other'

cat_stats = defaultdict(lambda: {'n': 0, 'our': 0, 'lvr': 0, 'pr': 0})
for id_, e in our.items():
    if id_ not in lvr or id_ not in pr: continue
    q_text = mmvp_q.get(id_, {}).get('Question', '')
    cat    = categorize(q_text)
    cat_stats[cat]['n']  += 1
    if acc_our(e['prediction'][0], e['label']):      cat_stats[cat]['our'] += 1
    if acc_lvr(lvr[id_]['prediction'][0], e['label']): cat_stats[cat]['lvr'] += 1
    if extract_pr(pr[id_]['prediction']) == e['label']: cat_stats[cat]['pr'] += 1

print(f"\n{'─'*80}")
print("Per-question-type accuracy:")
print(f"  {'Type':<20} {'our':>7} {'lvr':>7} {'pr':>7} {'n':>5}")
for cat, s in sorted(cat_stats.items()):
    print(f"  {cat:<20} {s['our']/s['n']:>6.1%} {s['lvr']/s['n']:>6.1%} {s['pr']/s['n']:>6.1%} {s['n']:>5}")

print(f"\n{'─'*80}")
print("PR wins — PR correct, we wrong:")
for e in pr_wins[:10]:
    print(f"\n  id={e['id']} label={e['label']} our='{e['our']}' pr='{e['pr']}'")
    print(f"  Q: {e['q']}  Options: {e['opts']}")

print(f"\n{'─'*80}")
print("Our wins — we correct, PR wrong:")
for e in our_wins[:10]:
    print(f"\n  id={e['id']} label={e['label']} our='{e['our']}' pr='{e['pr']}'")
    print(f"  Q: {e['q']}  Options: {e['opts']}")
