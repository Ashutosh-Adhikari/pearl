"""
Wins analysis: cases where PEARL is correct but BOTH LVR and PR are wrong.

Covers all three benchmarks (BLINK, VSTAR, MMVP) with question text.

Run with: python analysis/07_wins_analysis.py
"""
import json, csv
from pathlib import Path
from collections import Counter, defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (

    PEARL_BLINK, PEARL_VSTAR, PEARL_MMVP,
    LVR_BLINK, LVR_VSTAR, LVR_MMVP,
    PR_BLINK, PR_VSTAR, PR_MMVP,
    MMVP_CSV,
    acc_our, acc_lvr, extract_pr, pred_our,
)
from datasets import load_dataset


def load(path):
    with open(path) as f:
        return {d['id']: d for d in json.load(f)}


# ── Load all result files ─────────────────────────────────────────────────
our_blink = load(PEARL_BLINK)
our_vstar = load(PEARL_VSTAR)
our_mmvp  = load(PEARL_MMVP)
lvr_blink = load(LVR_BLINK)
lvr_vstar = load(LVR_VSTAR)
lvr_mmvp  = load(LVR_MMVP)
pr_blink  = load(PR_BLINK)
pr_vstar  = load(PR_VSTAR)
pr_mmvp   = load(PR_MMVP)

# ── Load question text ────────────────────────────────────────────────────
blink_cats = ['Counting', 'IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
blink_ds = {}
for cat in blink_cats:
    for d in load_dataset("BLINK-Benchmark/BLINK", cat)['val']:
        blink_ds[d['idx']] = d

vstar_ds = {d['question_id']: d for d in load_dataset("craigwu/vstar_bench")['test']}

mmvp_q = {}
with open(MMVP_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        mmvp_q[int(row['Index'])] = row


def find_wins(our, lvr, pr, question_fn, *, acc_lvr_fn=acc_lvr):
    """
    Returns list of cases where we are correct but BOTH LVR and PR are wrong.
    We use strict pred_our == label to avoid false positives from lenient substring match.
    question_fn(id_) → dict with keys 'q' (question str) and optionally 'opts'.
    """
    wins = []
    for id_, e in our.items():
        if pred_our(e['prediction'][0]) != e['label']:  # strict exact match
            continue
        if id_ not in lvr or id_ not in pr:
            continue
        lc  = acc_lvr_fn(lvr[id_]['prediction'][0], e['label'])
        prc = extract_pr(pr[id_]['prediction']) == e['label']
        if lc or prc:
            continue  # not a unique win
        lvr_p = pred_our(lvr[id_]['prediction'][0])
        pr_p  = extract_pr(pr[id_]['prediction'])
        qdat  = question_fn(id_)
        wins.append({
            'id': id_, 'label': e['label'],
            'our': pred_our(e['prediction'][0]),
            'lvr': lvr_p, 'pr': pr_p,
            'q':   qdat.get('q', ''),
            'opts': qdat.get('opts', ''),
            'cat': e.get('category', '?'),
        })
    return wins


def blink_qfn(id_):
    num = int(id_.split('_')[-1]) if '_' in str(id_) else id_
    dat = blink_ds.get(id_, blink_ds.get(num, {}))
    return {'q': dat.get('question', ''), 'opts': dat.get('choices', [])}

def vstar_qfn(id_):
    dat = vstar_ds.get(id_, {})
    return {'q': dat.get('text', ''), 'opts': ''}

def mmvp_qfn(id_):
    row = mmvp_q.get(id_, {})
    return {'q': row.get('Question', ''), 'opts': row.get('Options', '')}


blink_wins = find_wins(our_blink, lvr_blink, pr_blink, blink_qfn)
vstar_wins = find_wins(our_vstar, lvr_vstar, pr_vstar, vstar_qfn)
mmvp_wins  = find_wins(our_mmvp,  lvr_mmvp,  pr_mmvp,  mmvp_qfn)
# Note: pred_our (strict) is used above, which may give slightly lower counts than
# acc_our (lenient substring) used in the overall scoring — this is intentional to
# avoid false positives in the wins analysis.

# ── Per-category breakdowns ───────────────────────────────────────────────
blink_by_cat = defaultdict(list)
for w in blink_wins:
    blink_by_cat[w['cat']].append(w)

blink_total = {cat: sum(1 for _, e in our_blink.items() if e.get('category') == cat)
               for cat in blink_cats}
vstar_total = {cat: sum(1 for _, e in our_vstar.items() if e.get('category') == cat)
               for cat in ['direct_attributes', 'relative_position']}
vstar_by_cat = defaultdict(list)
for w in vstar_wins:
    vstar_by_cat[w['cat']].append(w)

print("=" * 80)
print("WINS ANALYSIS — PEARL correct, BOTH LVR and PR wrong")
print("=" * 80)
print(f"\nBLINK unique wins:  {len(blink_wins)} / {len(our_blink)}")
print(f"VSTAR unique wins:  {len(vstar_wins)} / {len(our_vstar)}")
print(f"MMVP  unique wins:  {len(mmvp_wins)}  / {len(our_mmvp)}")

# ── BLINK ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print("BLINK — unique wins per category")
print(f"{'─'*80}")
for cat in blink_cats:
    ws = blink_by_cat[cat]
    n  = blink_total[cat]
    print(f"\n  {cat} ({len(ws)} unique wins out of {n})")
    if not ws:
        continue
    # prediction distributions among wins
    lvr_pred_dist = Counter(w['lvr'] for w in ws)
    pr_pred_dist  = Counter(w['pr']  for w in ws)
    true_dist     = Counter(w['label'] for w in ws)
    print(f"    truth: {dict(sorted(true_dist.items()))}  lvr wrong→{dict(sorted(lvr_pred_dist.items()))}  pr wrong→{dict(sorted(pr_pred_dist.items()))}")
    for w in ws[:6]:
        print(f"\n    {w['id']}  truth={w['label']}  our='{w['our']}'  lvr='{w['lvr']}'  pr='{w['pr']}'")
        print(f"    Q: {str(w['q'])[:120]}")
        if w['opts']:
            print(f"    Opts: {w['opts']}")

# ── VSTAR ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print("VSTAR — unique wins per category")
print(f"{'─'*80}")
for cat in ['direct_attributes', 'relative_position']:
    ws = vstar_by_cat[cat]
    n  = vstar_total[cat]
    print(f"\n  {cat} ({len(ws)} unique wins out of {n})")
    if not ws:
        continue
    lvr_pd = Counter(w['lvr'] for w in ws)
    pr_pd  = Counter(w['pr']  for w in ws)
    true_d = Counter(w['label'] for w in ws)
    print(f"    truth: {dict(sorted(true_d.items()))}  lvr→{dict(sorted(lvr_pd.items()))}  pr→{dict(sorted(pr_pd.items()))}")
    for w in ws:
        print(f"\n    id={w['id']}  truth={w['label']}  our='{w['our']}'  lvr='{w['lvr']}'  pr='{w['pr']}'")
        print(f"    Q: {str(w['q'])[:150]}")

# ── MMVP ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print("MMVP — unique wins (we correct, both wrong)")
print(f"{'─'*80}")
true_d = Counter(w['label'] for w in mmvp_wins)
lvr_pd = Counter(w['lvr']   for w in mmvp_wins)
pr_pd  = Counter(w['pr']    for w in mmvp_wins)
print(f"  truth: {dict(sorted(true_d.items()))}  lvr→{dict(sorted(lvr_pd.items()))}  pr→{dict(sorted(pr_pd.items()))}")
for w in mmvp_wins[:12]:
    print(f"\n  id={w['id']}  truth={w['label']}  our='{w['our']}'  lvr='{w['lvr']}'  pr='{w['pr']}'")
    print(f"  Q: {str(w['q'])[:120]}")
    if w['opts']:
        print(f"  Opts: {str(w['opts'])[:80]}")

# ── MMVP question-type breakdown for wins ─────────────────────────────────
def categorize(q_text):
    q = q_text.lower()
    if any(k in q for k in ['shadow', 'reflect', 'light', 'bright', 'darker', 'lighter', 'illuminat']): return 'shadow/light'
    if any(k in q for k in ['left', 'right', 'facing', 'direction', 'toward', 'away']):                  return 'orientation'
    if any(k in q for k in ['visible', 'see', 'show', 'appear', 'entire', 'submerged']):                 return 'visibility'
    if any(k in q for k in ['how many', 'count', 'number', 'range']):                                    return 'counting'
    return 'other'

win_cats = Counter(categorize(w['q']) for w in mmvp_wins)
print(f"\n  MMVP wins by question type: {dict(win_cats)}")
