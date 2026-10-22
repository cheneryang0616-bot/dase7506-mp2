"""Grouped evaluation, paired bootstrap and error-case sampling for MP2.

Usage (from code/):
  ./.venv/bin/python ../tools/analyze.py \
      --checkpoint ../release/checkpoint/model_bundle \
      --out ../runs/analysis_model.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
sys.path.insert(0, str(HERE.parent / 'method'))

from common import load_inputs, read_jsonl, parse_answer  # noqa: E402
from submission import predict  # noqa: E402


def labels(bank, gold):
    y = np.zeros((len(bank), 4), dtype=np.int64)
    parsed_ref = [parse_answer(g['answer']) for g in gold]
    for q, (row, pref) in enumerate(zip(bank, parsed_ref)):
        for j, c in enumerate(row['candidates']):
            p = parse_answer(c['text'])
            y[q, j] = int(p is not None and p == pref)
    return y, parsed_ref


def majority_choice(bank):
    out = []
    for row in bank:
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        counts = Counter(p for p in parsed if p is not None)
        if not counts:
            out.append(0)
            continue
        top = max(counts.values())
        winners = {a for a, c in counts.items() if c == top}
        pick = next(j for j, p in enumerate(parsed) if p in winners)
        out.append(pick)
    return np.array(out)


def pattern_of(row):
    parsed = [parse_answer(c['text']) for c in row['candidates']]
    vals = [p for p in parsed if p is not None]
    fails = 4 - len(vals)
    if not vals:
        return 'all-fail'
    sizes = sorted(Counter(vals).values(), reverse=True)
    tag = '+'.join(map(str, sizes))
    return tag + (f'+{fails}fail' if fails else '')


def paired_bootstrap(y, pick_a, pick_b, n_boot=5000, seed=17):
    n = len(pick_a)
    a = np.array([y[q, pick_a[q]] for q in range(n)], dtype=float)
    b = np.array([y[q, pick_b[q]] for q in range(n)], dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (a[idx] - b[idx]).mean() * 100
    return {
        'delta_pct': round(100 * (a.mean() - b.mean()), 3),
        'ci95': [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)],
        'p_model_better': round(float((diffs > 0).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--out', type=Path, default=HERE.parent / 'runs/analysis_model.json')
    ap.add_argument('--cases', type=int, default=20)
    args = ap.parse_args()

    bank, arrays = load_inputs(args.split)
    gold = read_jsonl(HERE.parent / f'code/data/gsm8k_{args.split}.jsonl')
    y, _ = labels(bank, gold)
    n = len(bank)

    picks = np.array(predict(bank, arrays, args.checkpoint))
    mv = majority_choice(bank)
    first = np.zeros(n, dtype=int)
    model_acc = 100 * y[np.arange(n), picks].mean()
    mv_acc = 100 * y[np.arange(n), mv].mean()
    first_acc = 100 * y[np.arange(n), first].mean()

    patterns = [pattern_of(r) for r in bank]
    def acc_by(pick):
        d = {}
        for pat in sorted(set(patterns)):
            m = np.array([p == pat for p in patterns])
            d[pat] = {'n': int(m.sum()),
                      'model': round(100 * y[m, pick[m]].mean(), 2),
                      'majority': round(100 * y[m, mv[m]].mean(), 2)}
        return d

    # hard subset: >=2 distinct valid parsed answers
    hard = np.array([len({parse_answer(c['text']) for c in r['candidates']} - {None}) >= 2 for r in bank])
    soft = ~hard

    result = {
        'split': args.split, 'n': n,
        'model_accuracy_pct': round(model_acc, 3),
        'majority_accuracy_pct': round(mv_acc, 3),
        'first_accuracy_pct': round(first_acc, 3),
        'delta_model_vs_majority': paired_bootstrap(y, picks, mv),
        'delta_model_vs_first': paired_bootstrap(y, picks, first),
        'hard_subset': {
            'n': int(hard.sum()),
            'model': round(100 * y[hard, picks[hard]].mean(), 2),
            'majority': round(100 * y[hard, mv[hard]].mean(), 2),
        },
        'easy_subset': {
            'n': int(soft.sum()),
            'model': round(100 * y[soft, picks[soft]].mean(), 2),
            'majority': round(100 * y[soft, mv[soft]].mean(), 2),
        },
        'by_pattern': acc_by(picks),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'by_pattern'}, indent=2))

    # error-case sampling, fixed rule: majority right & model wrong, then both wrong, then model right & majority wrong
    cat = {
        'fixed_by_model': [q for q in range(n) if y[q, mv[q]] == 0 and y[q, picks[q]] == 1],
        'broken_by_model': [q for q in range(n) if y[q, mv[q]] == 1 and y[q, picks[q]] == 0],
        'both_wrong': [q for q in range(n) if y[q, mv[q]] == 0 and y[q, picks[q]] == 0],
    }
    rng = np.random.default_rng(42)
    cases = {}
    for k, idxs in cat.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        cases[k] = {'total': len(idxs), 'ids': [bank[q]['id'] for q in idxs[:args.cases]]}
    (args.out.parent / f'error_categories_{args.split}.json').write_text(json.dumps(cases, indent=2))
    print('\n' + json.dumps({k: v['total'] for k, v in cases.items()}, indent=2))


if __name__ == '__main__':
    main()
