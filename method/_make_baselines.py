"""Baselines B0-B3 for DASE7506 MP2. Each file is a standalone --method target."""

# ---------------------------------------------------------------- B0: random
B0 = '''"""B0: uniform random over 0..3 with fixed seed 42."""
import numpy as np


def predict(bank, arrays, checkpoint):
    rng = np.random.default_rng(42)
    return [int(rng.integers(0, 4)) for _ in range(len(bank))]
'''

# ---------------------------------------------------------------- B1: first
B1 = '''"""B1: always choose candidate 0."""


def predict(bank, arrays, checkpoint):
    return [0 for _ in range(len(bank))]
'''

# ---------------------------------------------------------------- B2: majority
B2 = '''"""B2: majority vote over parseable final answers; ties -> smallest original index; none parseable -> 0."""
from collections import Counter
from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []
    for row in bank:
        parsed = [parse_answer(c["text"]) for c in row["candidates"]]
        counts = Counter(p for p in parsed if p is not None)
        if not counts:
            choices.append(0)
            continue
        top = max(counts.values())
        winners = {a for a, c in counts.items() if c == top}
        pick = 0
        for j, p in enumerate(parsed):
            if p in winners:
                pick = j
                break
        choices.append(pick)
    return choices
'''

# ---------------------------------------------------------------- B3: first parseable
B3 = '''"""B3: first candidate whose final answer parses; else 0."""
from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []
    for row in bank:
        pick = 0
        for j, c in enumerate(row["candidates"]):
            if parse_answer(c["text"]) is not None:
                pick = j
                break
        choices.append(pick)
    return choices
'''

if __name__ == '__main__':
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for name, src in (('b0_random', B0), ('b1_first', B1), ('b2_majority', B2), ('b3_first_parseable', B3)):
        (here / f'{name}.py').write_text(src)
        print('wrote', name)
