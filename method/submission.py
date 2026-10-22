"""Frozen MP2 selection method: NumPy inference for the trained linear model.

predict(bank, arrays, checkpoint) -> list of Python ints in 0..3, bank order.

The checkpoint directory must contain `config.json` and `linear.npz` written by
method/train.py. Inference uses only the inputs passed in (question, candidate
texts, frozen embeddings); it never opens reference answers or question ids.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from features import FeatureSpec  # noqa: E402

TOL = 1e-9   # score tolerance for the pre-registered smallest-index tie rule


class Bundle:
    def __init__(self, checkpoint):
        path = Path(checkpoint)
        cfg_path, npz_path = path / 'config.json', path / 'linear.npz'
        if not npz_path.exists() or not cfg_path.exists():
            raise FileNotFoundError(
                f'Checkpoint bundle incomplete at {path}: need config.json and linear.npz')
        cfg = json.loads(cfg_path.read_text())
        with np.load(npz_path) as f:
            self.coef = f['coef'].astype(np.float64)
            self.intercept = float(f['intercept'][0])
            self.mean = f['mean'].astype(np.float64)
            self.std = f['std'].astype(np.float64)
        self.spec = FeatureSpec(
            emb_mode=cfg['emb_mode'], use_relative=cfg['use_relative'],
            use_cos=cfg['use_cos'], use_vote=cfg['use_vote'], use_surface=cfg['use_surface'])


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('This method needs --checkpoint pointing at the model bundle')
    bundle = Bundle(checkpoint)
    emb = arrays['embeddings']
    n = len(bank)
    if emb.shape != (n, 4, 1024):
        raise ValueError(f'Unexpected embedding shape {emb.shape} for {n} questions')
    X, _ = bundle.spec.blocks(bank, emb)
    flat = X.reshape(n * 4, X.shape[2])
    if flat.shape[1] != bundle.mean.shape[0]:
        raise ValueError(f'Feature dimension {flat.shape[1]} != bundle {bundle.mean.shape[0]}')
    z = (flat - bundle.mean) / bundle.std
    scores = (z @ bundle.coef + bundle.intercept).reshape(n, 4)
    choices = []
    for q in range(n):
        # Pre-registered tie rule: among candidates within TOL of the maximum score,
        # take the smallest original index. TOL absorbs float noise, so duplicated
        # candidates (identical score) always resolve to the same index and repeated
        # runs / peer reproduction agree bit-for-bit.
        row = scores[q]
        top = row.max()
        pick = int(np.flatnonzero(row >= top - TOL)[0])
        choices.append(pick)
    return choices
