"""Build the report tables from experiments.csv and the saved score JSONs."""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / 'runs'
RELEASE = ROOT / 'release'


def read_csv():
    with (ROOT / 'experiments.csv').open() as f:
        return list(csv.DictReader(f))


def best_per_config(rows, key='val_acc'):
    out = {}
    for r in rows:
        if not r.get('C'):
            continue
        name = r['config']
        v = float(r[key])
        if name not in out or float(out[name][key]) < v:
            out[name] = r
    return out


def score(path):
    return json.loads(Path(path).read_text())


def main():
    rows = read_csv()
    best = best_per_config(rows)
    L = []
    L.append('# MP2 result tables (auto-generated from experiments.csv and official scorers)\n')

    L.append('## Table 1 — Baselines and frozen model (official scorer)\n')
    L.append('| method | validation acc % | test acc % | note |')
    L.append('|---|---:|---:|---|')
    names = [('b0_random', 'B0 uniform random (seed 42)'),
             ('b1_first', 'B1 always candidate 0'),
             ('b2_majority', 'B2 majority vote over parsed answers'),
             ('b3_first_parseable', 'B3 first parseable candidate')]
    for key, desc in names:
        v = score(RUNS / f'validation/{key}-score.json')
        t = score(RUNS / f'test-{key}-score.json')
        L.append(f'| {desc} | {v["accuracy"]:.3f} | {t["accuracy"]:.3f} | fixed rule |')
    mv = score(RELEASE / 'test-score.json')
    vv = score(RUNS / 'validation/model-score.json')
    L.append(f'| **Frozen model (relative+vote, C=0.002)** | **{vv["accuracy"]:.3f}** | **{mv["accuracy"]:.3f}** | linear, dim 1029 |')
    L.append('')

    L.append('## Table 2 — Feature-group comparison (validation, best C per row)\n')
    L.append('| config | features | dim | best C | val acc % | train acc % |')
    L.append('|---|---|---:|---:|---:|---:|')
    order = ['m0_voteonly', 'm0_surfonly', 'm1_raw', 'm1_l2', 'm2_cos', 'm2_rel',
             'm2_vote', 'ablate_norel', 'rel_only_vote', 'rel_cos_vote', 'm2_full',
             'rel_surf_vote', 'm3_rel_surf', 'm4_full']
    for cfg in order:
        if cfg not in best:
            continue
        r = best[cfg]
        L.append(f'| {cfg} | {r["spec"]} | {r["dim"]} | {r["C"]} | {float(r["val_acc"]):.3f} | {float(r["train_acc"]):.3f} |')
    L.append('')

    L.append('## Table 3 - Direct feature ablations (fixed C=0.002, seed=42; validation)\n')
    L.append('These post-audit controls were run after the test result was known. They explain the frozen model; they do not select a replacement.\n')
    L.append('| comparison | features | dim | val acc % | delta vs frozen |')
    L.append('|---|---|---:|---:|---:|')
    full = float(best['rel_only_vote']['val_acc'])
    L.append(f'| frozen | rel+vote | 1029 | {full:.3f} | +0.000 |')
    for name, desc in [('exp_rel_only_C0.002_s42.json', 'remove vote'),
                       ('exp_vote_only_C0.002_s42.json', 'remove relative vector')]:
        r = score(RUNS / 'corrections' / name)
        a = r['validation_accuracy_pct']
        L.append(f'| {desc} | {r["spec"]} | {r["dim"]} | {a:.3f} | {a-full:+.3f} |')
    L.append('\nTable 2 is a feature-combination comparison with independently selected C, not a drop-one-group ablation.')
    L.append('')

    L.append('## Table 4 — Regularisation sensitivity of the frozen feature set (rel+vote)\n')
    L.append('| C | val acc % (seed 17/42/2026) |')
    L.append('|---|---|')
    seen = {}
    for r in rows:
        if r['config'] == 'rel_only_vote' and r.get('C'):
            seen.setdefault(r['C'], {})[r['seed']] = float(r['val_acc'])
    for C in sorted(seen, key=float):
        vals = [seen[C][s] for s in ('17', '42', '2026') if s in seen[C]]
        L.append(f'| {C} | ' + ' / '.join(f'{v:.3f}' for v in vals) + ' |')
    L.append('')

    L.append('## Table 5 — Nonlinear and group-loss alternatives (validation)\n')
    L.append('| method | val acc % | note |')
    L.append('|---|---:|---|')
    for r in rows:
        if not r.get('C') and 'mlp' in r['run_id']:
            L.append(f'| MLP hidden={r["dim"].split("mlp")[-1]} | {float(r["val_acc"]):.3f} | adam, 400 iters |')
    g = score(RUNS / 'corrections/exp_group_relvote_corrected.json')
    L.append(f'| corrected group multi-positive loss | {g["validation_accuracy_pct"]:.3f} | '
             f'group-mean gradient; best validation iteration {g["best_iter"]}; post-audit control |')
    L.append(f'| linear logreg (frozen) | {full:.3f} | reference |')
    L.append('\nThe historical group result in runs/exp_group_relvote.json is retained as a record of an incorrect gradient implementation, not evidence about the stated objective. MLP train accuracy is question-level selection accuracy.')
    L.append('')

    out = ROOT / 'report/tables.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
