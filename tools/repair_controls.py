"""Post-audit controls using training/validation only; never select a release model.

Run from the repository root:
  code/.venv/bin/python tools/repair_controls.py

Outputs are separate from the historical search and frozen release. Use --out
with a new directory for reproduction; existing experiment logs are not replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Enforce one compute thread, including children, before importing NumPy/BLAS.
THREAD_ENV = {
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1',
}
os.environ.update(THREAD_ENV)
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def mechanism_check():
    """Frozen linear scorer: subtracting the same group mean cancels in ranking."""
    import numpy as np
    sys.path.insert(0, str(ROOT / 'method'))
    sys.path.insert(0, str(ROOT / 'code'))
    from common import load_inputs
    from features import l2_normalize, vote_features

    bank, arrays = load_inputs('validation')  # input-only; no references needed
    checkpoint = ROOT / 'release/checkpoint/model_bundle'
    with np.load(checkpoint / 'linear.npz', allow_pickle=False) as b:
        w, mu, sd = b['coef'], b['mean'], b['std']
        intercept = float(b['intercept'][0])
    x = l2_normalize(arrays['embeddings'].astype(np.float64))
    group_mean = x.mean(axis=1, keepdims=True)
    vote = vote_features(bank)
    relative = np.concatenate([x - group_mean, vote], axis=-1)
    absolute = np.concatenate([x, vote], axis=-1)
    rel_scores = ((relative - mu) / sd) @ w + intercept
    abs_scores = ((absolute - mu) / sd) @ w + intercept
    delta = np.sum(group_mean[:, 0] * (w[:1024] / sd[:1024]), axis=1)
    error = float(np.max(np.abs(abs_scores - rel_scores - delta[:, None])))
    centered_error = float(np.max(np.abs(
        (abs_scores - abs_scores[:, :1]) - (rel_scores - rel_scores[:, :1]))))
    equal = np.array_equal(rel_scores.argmax(axis=1), abs_scores.argmax(axis=1))
    if not equal or centered_error >= 1e-10:
        raise AssertionError('Within-question ranking equivalence failed')
    return {
        'scope': 'validation inputs only; frozen release weights; no refitting or selection',
        'questions': len(bank), 'same_argmax_questions': int(equal) * len(bank),
        'max_abs_constant_shift_residual': error,
        'max_abs_pairwise_score_difference_residual': centered_error,
        'interpretation': (
            'For these fixed linear weights, replacing the centered embedding by '
            'its absolute L2-normalized embedding and keeping the same scaling '
            'adds a question-specific constant. It preserves candidate ranking. '
            'Retraining can still change scaling, regularization, and weights. '
            'Vote features continue to depend on other candidates.'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/corrections')
    args = ap.parse_args()
    out = args.out.resolve()
    names = ('gradient_test.log', 'exp_group_relvote_corrected.json',
             'exp_rel_only_C0.002_s42.json', 'exp_vote_only_C0.002_s42.json',
             'relative_linear_mechanism.json', 'repair_controls_manifest.json')
    if any((out / name).exists() for name in names):
        raise FileExistsError('Choose a new --out directory; correction records are append-only')
    out.mkdir(parents=True, exist_ok=True)

    frozen_paths = sorted((ROOT / 'release/checkpoint').rglob('*'))
    frozen_paths = [p for p in frozen_paths if p.is_file()]
    history = ROOT / 'runs/exp_group_relvote.json'
    before = {str(p.relative_to(ROOT)): digest(p) for p in [*frozen_paths, history]}
    manifest = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Post-audit mathematical correction and symmetric feature controls',
        'selection_policy': ('These results do not reselect, refit, or replace the final '
                             'release. The frozen C=0.002, seed=42 controls are diagnostic.'),
        'data_scope': 'Only train/validation inputs and references; mechanism check uses validation inputs only',
        'thread_limits': THREAD_ENV, 'python': sys.version, 'platform': platform.platform(),
        'historical_seed_note': ('Historical group command omitted --seed; default was 42. '
                                 'All-zero full-batch Adam initialization is deterministic.'),
        'frozen_and_historical_sha256_before': before,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (
            ROOT / 'method/train_group.py', ROOT / 'method/train.py',
            ROOT / 'tools/test_group_gradient.py', Path(__file__).resolve())},
        'runs': [],
    }

    tasks = [
        ('gradient_test', ['tools/test_group_gradient.py']),
        ('group_relvote_corrected', [
            'method/train_group.py', '--config', 'rel_only_vote', '--l2', '0.001',
            '--lr', '0.05', '--iters', '1500', '--seed', '42',
            '--bundle', str(out / 'bundles/group_relvote_corrected'),
            '--report', str(out / 'exp_group_relvote_corrected.json')]),
        ('rel_only_C0.002_s42', [
            'method/train.py', '--config', 'rel_only', '--C', '0.002', '--seed', '42',
            '--bundle', str(out / 'bundles/rel_only_C0.002_s42'),
            '--report', str(out / 'exp_rel_only_C0.002_s42.json')]),
        ('vote_only_C0.002_s42', [
            'method/train.py', '--config', 'm0_voteonly', '--C', '0.002', '--seed', '42',
            '--bundle', str(out / 'bundles/vote_only_C0.002_s42'),
            '--report', str(out / 'exp_vote_only_C0.002_s42.json')]),
    ]
    for name, argv in tasks:
        command = [sys.executable, *argv]
        print(f'Running {name}', flush=True)
        t0 = time.perf_counter()
        with (out / f'{name}.log').open('w') as log:
            cp = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                stdout=log, stderr=subprocess.STDOUT, check=False)
        record = {'name': name, 'argv': command, 'returncode': cp.returncode,
                  'wall_seconds': round(time.perf_counter() - t0, 6),
                  'log': str((out / f'{name}.log').relative_to(ROOT))}
        manifest['runs'].append(record)
        if cp.returncode:
            write_json(out / 'repair_controls_manifest.json', manifest)
            raise RuntimeError(f'{name} failed; see its log')

    write_json(out / 'relative_linear_mechanism.json', mechanism_check())
    after = {str(p.relative_to(ROOT)): digest(p) for p in [*frozen_paths, history]}
    if before != after:
        raise AssertionError('Frozen release or historical group JSON changed')
    manifest['frozen_and_historical_sha256_after'] = after
    manifest['frozen_and_historical_unchanged'] = True
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        from threadpoolctl import threadpool_info
        manifest['effective_threadpools'] = threadpool_info()
    except ImportError:
        manifest['effective_threadpools'] = 'threadpoolctl unavailable; environment recorded'
    write_json(out / 'repair_controls_manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
