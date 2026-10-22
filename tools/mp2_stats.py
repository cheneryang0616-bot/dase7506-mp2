"""Phase-2 development statistics for DASE7506 MP2.

Reads only the supplied inputs and references for train/validation.
Produces a JSON + Markdown profile of:
  - label generation (y[q,j] via the official parse_answer)
  - per-question correct-candidate counts and per-position accuracy
  - parseability / distinct-valid-answer distributions
  - answer vote patterns
  - surface features (length, digits, lines, answer markers) vs correctness
  - majority-vote coverage and its correctness
  - the "hard" validation subset: >=2 distinct valid parsed answers

Run from code/ with the project venv:
  python ../tools/mp2_stats.py --out ../runs/phase2_stats.json
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'code'
sys.path.insert(0, str(CODE))

from common import load_inputs, read_jsonl, parse_answer  # noqa: E402

SPLITS = ('train', 'validation')


def labels_for(bank, gold):
    """y[q][j] = 1 iff candidate parses and equals the parsed reference."""
    parsed_ref = [parse_answer(g['answer']) for g in gold]
    Y, P, Q = [], [], []
    for row, pref in zip(bank, parsed_ref):
        pc = [parse_answer(c['text']) for c in row['candidates']]
        P.append(pc)
        Q.append(pref)
        Y.append([int(p is not None and p == pref) for p in pc])
    return Y, P, Q


def surface(text):
    digits = re.findall(r'\d+', text)
    return {
        'chars': len(text),
        'lines': text.count('\n') + 1,
        'digits': len(digits),
        'has_boxed': int('\\boxed' in text),
        'has_equals': int('=' in text),
    }


def vote_pattern(parsed):
    """Pattern string over parseable values; None votes are NOT pooled."""
    vals = [p for p in parsed if p is not None]
    if not vals:
        return 'all-fail'
    counts = Counter(vals)
    sizes = sorted(counts.values(), reverse=True)
    fails = len(parsed) - len(vals)
    tag = '+'.join(str(s) for s in sizes)
    if fails:
        tag += f' (+{fails}fail)'
    return tag


def majority_answer(parsed):
    """Return (answer, unique_winner) using only parseable values; ties -> smallest-index candidate's value."""
    counts = Counter(v for v in parsed if v is not None)
    if not counts:
        return None, False
    top = max(counts.values())
    winners = [a for a, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0], True
    # tie: pick the value of the smallest-index candidate among winners
    for p in parsed:
        if p in winners:
            return p, False
    return winners[0], False


def analyze(split, data_dir=None):
    bank, arrays = load_inputs(split) if data_dir is None else load_inputs(split, data_dir)
    gold = read_jsonl(CODE / f'data/gsm8k_{split}.jsonl')
    Y, P, Q = labels_for(bank, gold)
    n = len(bank)

    correct_counts = Counter(sum(y) for y in Y)
    pos_correct = [sum(y[j] for y in Y) for j in range(4)]
    parse_counts = Counter(sum(p is not None for p in pc) for pc in P)
    distinct_valid = Counter(len({p for p in pc if p is not None}) for pc in P)
    patterns = Counter(vote_pattern(pc) for pc in P)

    # surface vs correctness (pooled over candidate records)
    correct_surface, wrong_surface = [], []
    len_buckets = [(0, 200), (200, 400), (400, 700), (700, 10 ** 9)]
    len_bucket_stats = {f'{a}-{b}': {'correct': 0, 'total': 0} for a, b in len_buckets}
    boxed_stats = {'boxed': {'correct': 0, 'total': 0}, 'not_boxed': {'correct': 0, 'total': 0}}
    for row, y in zip(bank, Y):
        for c, yy in zip(row['candidates'], y):
            f = surface(c['text'])
            (correct_surface if yy else wrong_surface).append(f)
            for a, b in len_buckets:
                if a <= f['chars'] < b:
                    len_bucket_stats[f'{a}-{b}']['total'] += 1
                    len_bucket_stats[f'{a}-{b}']['correct'] += yy
                    break
            k = 'boxed' if f['has_boxed'] else 'not_boxed'
            boxed_stats[k]['total'] += 1
            boxed_stats[k]['correct'] += yy

    def mean(xs, k):
        return round(statistics.fmean(x[k] for x in xs), 2) if xs else None

    # majority vote outcome
    mv_correct = mv_wrong = mv_allfail = mv_tie = 0
    for row, y, pc in zip(bank, Y, P):
        ans, unique = majority_answer(pc)
        if ans is None:
            mv_allfail += 1
            continue
        if not unique:
            mv_tie += 1
        # would choosing the majority answer be right with a smallest-index tie-break?
        # (majority_answer already returns the tie-break value; correctness vs reference)
        # find a candidate with that value
        j = next(i for i, p in enumerate(pc) if p == ans)
        if y[j]:
            mv_correct += 1
        else:
            mv_wrong += 1

    hard = [i for i, pc in enumerate(P) if len({p for p in pc if p is not None}) >= 2]

    profile = {
        'split': split,
        'n_questions': n,
        'candidate_records': 4 * n,
        'mean_correct_per_question': round(statistics.fmean(sum(y) for y in Y), 4),
        'correct_count_distribution': {str(k): v for k, v in sorted(correct_counts.items())},
        'position_accuracy_counts': pos_correct,
        'position_accuracy_pct': [round(100 * c / n, 2) for c in pos_correct],
        'parseable_count_distribution': {str(k): v for k, v in sorted(parse_counts.items())},
        'distinct_valid_answer_distribution': {str(k): v for k, v in sorted(distinct_valid.items())},
        'vote_patterns': dict(patterns.most_common()),
        'majority_vote': {
            'unique_winner_correct': mv_correct,
            'unique_winner_wrong': mv_wrong,
            'tie_or_multi': mv_tie,
            'all_parse_fail': mv_allfail,
            'pct_correct_if_choose_majority': round(100 * mv_correct / n, 2),
        },
        'surface_pooled': {
            'correct': {'n': len(correct_surface), 'chars': mean(correct_surface, 'chars'),
                        'lines': mean(correct_surface, 'lines'), 'digits': mean(correct_surface, 'digits'),
                        'pct_boxed': round(100 * mean(correct_surface, 'has_boxed'), 1)},
            'wrong': {'n': len(wrong_surface), 'chars': mean(wrong_surface, 'chars'),
                      'lines': mean(wrong_surface, 'lines'), 'digits': mean(wrong_surface, 'digits'),
                      'pct_boxed': round(100 * mean(wrong_surface, 'has_boxed'), 1)},
        },
        'length_buckets': {k: {**v, 'pct_correct': round(100 * v['correct'] / v['total'], 2) if v['total'] else None}
                           for k, v in len_bucket_stats.items()},
        'boxed_vs_correct': {k: {**v, 'pct_correct': round(100 * v['correct'] / v['total'], 2) if v['total'] else None}
                             for k, v in boxed_stats.items()},
        'hard_subset_ge2valid': {'n': len(hard), 'pct': round(100 * len(hard) / n, 2)},
    }
    return profile, bank, Y, P


def render_md(profiles):
    out = ['# MP2 Phase-2 development profile', '']
    for p in profiles:
        out.append(f'## Split: {p["split"]}  (n={p["n_questions"]})')
        out.append('')
        out.append(f'- Mean correct candidates per question: **{p["mean_correct_per_question"]}**')
        out.append(f'- Correct-count distribution (0..4): {p["correct_count_distribution"]}')
        out.append(f'- Per-position accuracy %: {p["position_accuracy_pct"]}')
        out.append(f'- Parseable candidates per question: {p["parseable_count_distribution"]}')
        out.append(f'- Distinct valid answers per question: {p["distinct_valid_answer_distribution"]}')
        out.append(f'- Hard subset (>=2 distinct valid answers): {p["hard_subset_ge2valid"]}')
        mv = p['majority_vote']
        out.append(f'- Majority vote: correct={mv["unique_winner_correct"]} wrong={mv["unique_winner_wrong"]} '
                   f'tie={mv["tie_or_multi"]} all-fail={mv["all_parse_fail"]} '
                   f'-> **{mv["pct_correct_if_choose_majority"]}%**')
        out.append(f'- Surface means (correct vs wrong): {p["surface_pooled"]}')
        out.append(f'- Length buckets: {p["length_buckets"]}')
        out.append(f'- Boxed marker vs correctness: {p["boxed_vs_correct"]}')
        out.append('')
        out.append('Top vote patterns:')
        for k, v in list(p['vote_patterns'].items())[:12]:
            out.append(f'  - `{k}`: {v} ({round(100*v/p["n_questions"],2)}%)')
        out.append('')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE.parent / 'runs/phase2_stats.json')
    args = ap.parse_args()
    profiles = []
    for split in SPLITS:
        prof, bank, Y, P = analyze(split)
        profiles.append(prof)
        # dump a small deterministic sample of 20 train questions for manual reading
        if split == 'train':
            sample = []
            for i in range(20):
                sample.append({
                    'id': bank[i]['id'], 'question': bank[i]['question'],
                    'candidates': [c['text'] for c in bank[i]['candidates']],
                    'parsed': P[i], 'labels': Y[i],
                })
            args.out.parent.mkdir(parents=True, exist_ok=True)
            (args.out.parent / 'phase2_sample20.json').write_text(json.dumps(sample, indent=2, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({p['split']: p for p in profiles}, indent=2))
    md = render_md(profiles)
    (args.out.parent / 'phase2_stats.md').write_text(md)
    print(md)


if __name__ == '__main__':
    main()
