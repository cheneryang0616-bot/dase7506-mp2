"""In-process experiment runner: loads data once, sweeps configs/C/seeds.

Writes/append to experiments.csv and per-run JSON under runs/exp/.

Usage (from code/):
  ./.venv/bin/python ../tools/sweep.py --configs m2_vote,m2_full --C 0.01,0.1,1,10
  ./.venv/bin/python ../tools/sweep.py --configs m2_vote --C 1 --seeds 17,42,2026
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
sys.path.insert(0, str(HERE.parent / 'method'))

import train as T  # noqa: E402
from common import load_inputs, read_jsonl, parse_answer  # noqa: E402

ROOT = HERE.parent
CSV_PATH = ROOT / 'experiments.csv'
FIELDS = ['run_id', 'config', 'spec', 'dim', 'C', 'seed', 'train_acc', 'val_acc',
          'feature_seconds', 'fit_seconds', 'assumption', 'note']


def load_labels(split):
    bank, arrays = load_inputs(split)
    gold = read_jsonl(ROOT / f'code/data/gsm8k_{split}.jsonl')
    y = np.zeros((len(bank), 4), dtype=np.int64)
    for q, (row, g) in enumerate(zip(bank, gold)):
        pref = parse_answer(g['answer'])
        for j, c in enumerate(row['candidates']):
            p = parse_answer(c['text'])
            y[q, j] = int(p is not None and p == pref)
    return bank, arrays['embeddings'], y


def append_csv(rows):
    new = not CSV_PATH.exists()
    with CSV_PATH.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True)
    ap.add_argument('--C', default='1')
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--mlp', default='', help='comma list of hidden widths to also try')
    ap.add_argument('--note', default='')
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(',') if c.strip()]
    Cs = [float(c) for c in args.C.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    bank_tr, emb_tr, y_tr = load_labels('train')
    bank_va, emb_va, y_va = load_labels('validation')
    yflat = y_tr.reshape(-1)
    print(f'loaded train={len(bank_tr)} val={len(bank_va)} pos_rate={yflat.mean():.4f}', flush=True)

    rows, reports = [], []
    for cfg in configs:
        spec = T.CONFIGS[cfg]
        t0 = time.perf_counter()
        ztr, stats = T.build_matrix(spec, bank_tr, emb_tr)
        zva, _ = T.build_matrix(spec, bank_va, emb_va, stats)
        feat_s = round(time.perf_counter() - t0, 3)
        for C in Cs:
            for seed in seeds:
                t1 = time.perf_counter()
                coef, b, _ = T.fit_logreg(ztr, yflat, C, seed)
                fit_s = round(time.perf_counter() - t1, 3)
                acc_tr, _, _ = T.question_accuracy(ztr, y_tr, coef, b, len(bank_tr))
                acc_va, _, _ = T.question_accuracy(zva, y_va, coef, b, len(bank_va))
                rid = f'{cfg}_C{C}_s{seed}'
                rep = {'run_id': rid, 'config': cfg, 'spec': spec.name, 'dim': int(ztr.shape[1]),
                       'C': C, 'seed': seed, 'train_accuracy_pct': round(acc_tr, 4),
                       'validation_accuracy_pct': round(acc_va, 4)}
                (ROOT / 'runs').mkdir(exist_ok=True)
                (ROOT / f'runs/exp_{rid}.json').write_text(json.dumps(rep, indent=2))
                reports.append(rep)
                rows.append({'run_id': rid, 'config': cfg, 'spec': spec.name, 'dim': ztr.shape[1],
                             'C': C, 'seed': seed, 'train_acc': round(acc_tr, 3),
                             'val_acc': round(acc_va, 3), 'feature_seconds': feat_s,
                             'fit_seconds': fit_s, 'assumption': 'H1/H2',
                             'note': args.note})
                print(f'  {rid:32s} train={acc_tr:6.2f} val={acc_va:6.2f} fit={fit_s}s', flush=True)
                # keep best bundle later; save this one for reference
                T.save_bundle(ROOT / 'runs/bundles' / rid, coef, b, stats, spec, rep)

        if args.mlp:
            from sklearn.metrics import log_loss
            for hidden in [int(h) for h in args.mlp.split(',') if h.strip()]:
                t2 = time.perf_counter()
                clf = T.fit_mlp(ztr, yflat, hidden, seeds[0])
                fit_s = round(time.perf_counter() - t2, 3)
                pva = clf.predict_proba(zva)[:, 1].reshape(len(bank_va), 4)
                pick = pva.argmax(axis=1)
                acc_va = 100 * y_va[np.arange(len(bank_va)), pick].mean()
                ptra = clf.predict_proba(ztr)[:, 1].reshape(len(bank_tr), 4)
                acc_tr = 100 * y_tr[np.arange(len(bank_tr)), ptra.argmax(axis=1)].mean()
                rid = f'{cfg}_mlp{hidden}_s{seeds[0]}'
                rows.append({'run_id': rid, 'config': cfg, 'spec': spec.name,
                             'dim': f'{ztr.shape[1]}+mlp{hidden}', 'C': '', 'seed': seeds[0],
                             'train_acc': round(acc_tr, 3), 'val_acc': round(acc_va, 3),
                             'feature_seconds': feat_s, 'fit_seconds': fit_s,
                             'assumption': 'H1/H2-nonlinear', 'note': args.note})
                print(f'  {rid:32s} train={acc_tr:6.2f} val={acc_va:6.2f} fit={fit_s}s', flush=True)
                np.savez(ROOT / f'runs/bundles/{rid}.npz', spec=spec.name)

    append_csv(rows)
    print(f'\nappended {len(rows)} rows to experiments.csv')


if __name__ == '__main__':
    main()
