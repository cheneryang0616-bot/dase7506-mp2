"""Optional enhancement: group-level multi-positive loss (GUIDE section 6.4).

L(q) = -log( sum_{j in P(q)} softmax(s_q)[j] ) + (lambda/2)||w||^2

Questions whose candidates are all positive contribute zero gradient, which is
expected and recorded. Implemented with NumPy (Adam) so the same lightweight
inference bundle format can be reused.

Usage (from code/):
  ./.venv/bin/python ../method/train_group.py --config rel_only_vote \
      --l2 1e-3 --lr 0.05 --iters 400 --bundle ../runs/bundles/group_relvote
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'code'))

import train as T  # noqa: E402
from common import load_inputs, read_jsonl, parse_answer  # noqa: E402


def load(split):
    bank, arrays = load_inputs(split)
    gold = read_jsonl(HERE.parent / f'code/data/gsm8k_{split}.jsonl')
    y = np.zeros((len(bank), 4), dtype=np.float64)
    for q, (row, g) in enumerate(zip(bank, gold)):
        pref = parse_answer(g['answer'])
        for j, c in enumerate(row['candidates']):
            p = parse_answer(c['text'])
            y[q, j] = float(p is not None and p == pref)
    return bank, arrays['embeddings'], y


def group_loss_grad(Z, Y, lam, w):
    S = (Z @ w).reshape(-1, 4)
    if Y.shape != S.shape or not np.all((Y == 0) | (Y == 1)):
        raise ValueError('Y must be a binary [N, 4] array matching Z')
    if not np.all(Y.sum(axis=1) > 0):
        raise ValueError('group loss requires at least one positive per question')

    # Stable log-sum-exp over all candidates and over positives, respectively.
    # Their difference is -log(sum of positive softmax probabilities).
    shift = S.max(axis=1, keepdims=True)
    E = np.exp(S - shift)
    all_sum = E.sum(axis=1, keepdims=True)
    positive_scores = np.where(Y == 1, S, -np.inf)
    positive_shift = positive_scores.max(axis=1, keepdims=True)
    positive_exp = np.exp(positive_scores - positive_shift)
    positive_sum = positive_exp.sum(axis=1, keepdims=True)
    log_all = shift[:, 0] + np.log(all_sum[:, 0])
    log_positive = positive_shift[:, 0] + np.log(positive_sum[:, 0])
    loss = (log_all - log_positive).mean() + 0.5 * lam * float(w @ w)

    # The objective averages N QUESTIONS, not the 4N candidate rows.
    G = (E / all_sum - positive_exp / positive_sum) / Y.shape[0]
    grad = Z.T @ G.reshape(-1) + lam * w
    return loss, grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='rel_only_vote')
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bundle', type=Path, default=HERE.parent / 'runs/bundles/group_relvote')
    ap.add_argument('--report', type=Path, default=None)
    args = ap.parse_args()

    spec = T.CONFIGS[args.config]
    t_start = time.perf_counter()
    bank_tr, emb_tr, y_tr = load('train')
    bank_va, emb_va, y_va = load('validation')
    ztr, stats = T.build_matrix(spec, bank_tr, emb_tr)
    zva, _ = T.build_matrix(spec, bank_va, emb_va, stats)
    feature_s = time.perf_counter() - t_start

    all_pos = int((y_tr.sum(axis=1) == 4).sum())
    n = ztr.shape[1]
    w = np.zeros(n)
    m = np.zeros(n)
    v = np.zeros(n)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = (-1, None, None)
    checkpoints = []
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        loss, grad = group_loss_grad(ztr, y_tr, args.l2, w)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mh = m / (1 - b1 ** it)
        vh = v / (1 - b2 ** it)
        w = w - args.lr * mh / (np.sqrt(vh) + eps)
        if it % 50 == 0 or it == args.iters:
            acc_tr = T.question_accuracy(ztr, y_tr, w, 0.0, len(bank_tr))[0]
            acc_va = T.question_accuracy(zva, y_va, w, 0.0, len(bank_va))[0]
            if acc_va > best[0]:
                best = (acc_va, w.copy(), it)
            checkpoints.append({'iteration': it, 'loss_before_update': float(loss),
                                'train_accuracy_pct': float(acc_tr),
                                'validation_accuracy_pct': float(acc_va)})
            print(f'it={it:4d} loss={loss:.5f} train={acc_tr:6.2f} val={acc_va:6.2f}', flush=True)
    fit_s = time.perf_counter() - t0
    acc_va, wbest, it = best
    acc_tr = T.question_accuracy(ztr, y_tr, wbest, 0.0, len(bank_tr))[0]
    rep = {'config': args.config, 'loss': 'group-multi-positive',
           'all_positive_groups_train': all_pos, 'l2': args.l2, 'lr': args.lr,
           'iters': args.iters, 'best_iter': it, 'seed': args.seed,
           'initialization': 'all-zero weights; seed has no effect',
           'gradient_normalization': 'mean over N questions (corrected from 4N candidates)',
           'train_questions': len(bank_tr), 'validation_questions': len(bank_va),
           'train_accuracy_pct': round(acc_tr, 4),
           'validation_accuracy_pct': round(acc_va, 4), 'fit_seconds': round(fit_s, 3),
           'feature_seconds': round(feature_s, 3),
           'checkpoints': checkpoints,
           'dim': int(n)}
    if wbest is not None:
        T.save_bundle(args.bundle, wbest, 0.0, stats, spec, rep)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
