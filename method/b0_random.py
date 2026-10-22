"""B0: uniform random over 0..3 with fixed seed 42."""
import numpy as np


def predict(bank, arrays, checkpoint):
    rng = np.random.default_rng(42)
    return [int(rng.integers(0, 4)) for _ in range(len(bank))]
