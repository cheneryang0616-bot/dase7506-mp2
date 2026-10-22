"""Frozen-method contract checks (GUIDE section 9 "special checks").

Checks that the submitted predict() honours the interface for arbitrary bank
sizes, returns Python ints in 0..3 in bank order, is deterministic, and fails
clearly when the checkpoint is missing.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'method'))

from common import load_inputs  # noqa: E402
from submission import predict  # noqa: E402

CKPT = ROOT / 'release/checkpoint/model_bundle'
ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f'[{"PASS" if cond else "FAIL"}] {name}')


bank, arrays = load_inputs('validation')
emb = arrays['embeddings']

# arbitrary counts, including sizes that are not a multiple of the public splits
for k in (1, 5, 7, 100, 798):
    sub_bank = bank[:k]
    sub = {'embeddings': emb[:k]}
    out = predict(sub_bank, sub, CKPT)
    check(f'n={k}: one choice per question', len(out) == k)
    check(f'n={k}: python ints in 0..3', all(type(i) is int and 0 <= i < 4 for i in out))

# determinism
a = predict(bank, arrays, CKPT)
b = predict(bank, arrays, CKPT)
check('deterministic across repeated calls', a == b)

# bank order: reversing the bank must reverse the choices of an order-independent method
rev_bank = list(reversed(bank))
rev = predict(rev_bank, {'embeddings': emb[::-1]}, CKPT)
check('choices follow bank order', rev == list(reversed(a)))

# missing checkpoint -> clear error
try:
    predict(bank, arrays, ROOT / 'release/checkpoint/does_not_exist')
    check('missing checkpoint raises', False)
except FileNotFoundError as e:
    check('missing checkpoint raises FileNotFoundError', 'incomplete' in str(e))

# checkpoint=None -> clear error
try:
    predict(bank, arrays, None)
    check('None checkpoint raises', False)
except ValueError:
    check('None checkpoint raises ValueError', True)

# shape guard
try:
    predict(bank[:10], {'embeddings': emb[:9]}, CKPT)
    check('shape mismatch raises', False)
except ValueError:
    check('shape mismatch raises ValueError', True)

print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
