"""Train the MP2 selection model and write a lightweight inference bundle.

Usage (from code/):
  ./.venv/bin/python ../method/train.py --config m1_l2 --C 1.0 \
      --bundle ../release/checkpoint/model_bundle --report ../runs/exp_m1_l2.json

Only training inputs/references are used to fit. Validation is used for
comparison and reporting, never for fitting.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'code'))

from features import FeatureSpec, l2_normalize  # noqa: E402
from common import load_inputs, read_jsonl, parse_answer  # noqa: E402

# --------------------------------------------------------------- presets
CONFIGS = {
    'm0_voteonly':   FeatureSpec(use_vote=True),
    'm0_surfonly':   FeatureSpec(use_surface=True),
    'm1_raw':        FeatureSpec(emb_mode='raw'),
    'm1_l2':         FeatureSpec(emb_mode='l2'),
    'm2_rel':        FeatureSpec(emb_mode='l2', use_relative=True),
    'm2_cos':        FeatureSpec(emb_mode='l2', use_cos=True),
    'm2_vote':       FeatureSpec(emb_mode='l2', use_vote=True),
    'm2_full':       FeatureSpec(emb_mode='l2', use_relative=True, use_cos=True, use_vote=True),
    'm3_surf':       FeatureSpec(use_surface=True, use_vote=True),
    'm3_rel_surf':   FeatureSpec(emb_mode='l2', use_relative=True, use_cos=True, use_vote=True, use_surface=True),
    'ablate_norel':  FeatureSpec(emb_mode='l2', use_cos=True, use_vote=True),
    'ablate_novote': FeatureSpec(emb_mode='l2', use_relative=True, use_cos=True),
    'ablate_nosurf': FeatureSpec(emb_mode='l2', use_relative=True, use_cos=True, use_vote=True),
    'ablate_nocos':  FeatureSpec(emb_mode='l2', use_relative=True, use_vote=True),
    'rel_only_vote': FeatureSpec(use_relative=True, use_vote=True),
    'rel_only':      FeatureSpec(use_relative=True),
    'm2_relvote':    FeatureSpec(emb_mode='l2', use_relative=True, use_vote=True),
    'm4_full':       FeatureSpec(emb_mode='l2', use_relative=True, use_cos=True, use_vote=True, use_surface=True),
    'rel_cos_vote':  FeatureSpec(use_relative=True, use_cos=True, use_vote=True),
    'rel_surf_vote': FeatureSpec(use_relative=True, use_vote=True, use_surface=True),
}


def load_labels(split):
    bank, arrays = load_inputs(split)
    gold = read_jsonl(HERE.parent / f'code/data/gsm8k_{split}.jsonl')
    y = np.zeros((len(bank), 4), dtype=np.int64)
    parsed_ref = [parse_answer(g['answer']) for g in gold]
    # also cache parsed answers of candidates for analysis
    for q, (row, pref) in enumerate(zip(bank, parsed_ref)):
        for j, c in enumerate(row['candidates']):
            p = parse_answer(c['text'])
            y[q, j] = int(p is not None and p == pref)
    return bank, arrays['embeddings'], y, parsed_ref


def build_matrix(spec, bank, emb, stats=None):
    X, layout = spec.blocks(bank, emb)            # [N,4,d]
    N = X.shape[0]
    flat = X.reshape(N * 4, X.shape[2])
    if stats is None:
        mean = np.zeros(flat.shape[1])
        std = np.ones(flat.shape[1])
        for name, sl, needs in layout:
            if needs:
                blk = flat[:, sl]
                mean[sl] = blk.mean(axis=0)
                std[sl] = blk.std(axis=0)
                std[sl][std[sl] < 1e-8] = 1.0
        stats = {'mean': mean, 'std': std, 'layout': [(n, s.start, s.stop, ns) for n, s, ns in layout]}
    z = (flat - stats['mean']) / stats['std']
    return z, stats


def question_accuracy(z, y, coef, intercept, n_questions):
    scores = (z @ coef + intercept).reshape(n_questions, 4)
    pick = scores.argmax(axis=1)
    correct = sum(int(y[q, pick[q]] == 1) for q in range(n_questions))
    return 100.0 * correct / n_questions, pick, scores


def fit_logreg(z, yflat, C, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs', random_state=seed)
    clf.fit(z, yflat)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0]), clf


def fit_mlp(z, yflat, hidden, seed, alpha=1e-4):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=400,
                        random_state=seed, early_stopping=False, solver='adam',
                        batch_size=64, learning_rate_init=1e-3)
    clf.fit(z, yflat)
    return clf


def save_bundle(path, coef, intercept, stats, spec, meta):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / 'linear.npz',
             coef=np.asarray(coef, dtype=np.float64),
             intercept=np.asarray([intercept], dtype=np.float64),
             mean=np.asarray(stats['mean'], dtype=np.float64),
             std=np.asarray(stats['std'], dtype=np.float64))
    (path / 'config.json').write_text(json.dumps({
        'spec': spec.name,
        'emb_mode': spec.emb_mode,
        'use_relative': spec.use_relative,
        'use_cos': spec.use_cos,
        'use_vote': spec.use_vote,
        'use_surface': spec.use_surface,
        'layout': stats['layout'],
        'meta': meta,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=sorted(CONFIGS))
    ap.add_argument('--C', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bundle', type=Path, default=None)
    ap.add_argument('--report', type=Path, default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    spec = CONFIGS[args.config]
    t0 = time.perf_counter()
    bank_tr, emb_tr, y_tr, _ = load_labels('train')
    bank_va, emb_va, y_va, _ = load_labels('validation')
    ztr, stats = build_matrix(spec, bank_tr, emb_tr)
    zva, _ = build_matrix(spec, bank_va, emb_va, stats)
    yflat = y_tr.reshape(-1)
    t_feat = time.perf_counter()

    coef, intercept, clf = fit_logreg(ztr, yflat, args.C, args.seed)
    t_fit = time.perf_counter()

    acc_tr, _, _ = question_accuracy(ztr, y_tr, coef, intercept, len(bank_tr))
    acc_va, pick_va, _ = question_accuracy(zva, y_va, coef, intercept, len(bank_va))

    report = {
        'config': args.config, 'spec': spec.name, 'dim': int(ztr.shape[1]),
        'C': args.C, 'seed': args.seed,
        'train_accuracy_pct': round(acc_tr, 4),
        'validation_accuracy_pct': round(acc_va, 4),
        'feature_seconds': round(t_feat - t0, 3),
        'fit_seconds': round(t_fit - t_feat, 3),
        'train_pos_rate': round(float(yflat.mean()), 4),
    }
    if args.bundle:
        save_bundle(args.bundle, coef, intercept, stats, spec, report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
