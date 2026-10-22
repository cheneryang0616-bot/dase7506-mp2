"""Deterministic mathematical checks for the corrected group objective.

This uses synthetic arrays only; it reads no project split or checkpoint.
Run from the repository root with code/.venv/bin/python tools/test_group_gradient.py.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'method'))
from train_group import group_loss_grad  # noqa: E402


def finite_difference(z, y, lam, w, eps=1e-6):
    numerical = np.empty_like(w)
    for j in range(len(w)):
        plus, minus = w.copy(), w.copy()
        plus[j] += eps
        minus[j] -= eps
        numerical[j] = (group_loss_grad(z, y, lam, plus)[0]
                        - group_loss_grad(z, y, lam, minus)[0]) / (2 * eps)
    return numerical


class GroupGradientTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7506)
        self.z = rng.normal(size=(16, 7))
        self.w = rng.normal(size=7) * 0.25
        self.y = np.array([[1, 0, 0, 0], [1, 1, 0, 0],
                           [0, 1, 0, 1], [1, 1, 1, 1]], dtype=float)

    def check_gradient(self, y, lam):
        _, analytical = group_loss_grad(self.z, y, lam, self.w)
        numerical = finite_difference(self.z, y, lam, self.w)
        np.testing.assert_allclose(analytical, numerical, atol=2e-8, rtol=2e-6)
        return float(np.max(np.abs(analytical - numerical)))

    def test_multi_positive_without_l2(self):
        self.check_gradient(self.y, 0.0)

    def test_multi_positive_with_l2(self):
        self.check_gradient(self.y, 0.13)

    def test_all_positive_without_l2(self):
        y = np.ones_like(self.y)
        loss, grad = group_loss_grad(self.z, y, 0.0, self.w)
        self.assertEqual(loss, 0.0)
        np.testing.assert_array_equal(grad, np.zeros_like(self.w))
        self.check_gradient(y, 0.0)

    def test_all_positive_with_l2(self):
        y, lam = np.ones_like(self.y), 0.13
        loss, grad = group_loss_grad(self.z, y, lam, self.w)
        self.assertAlmostEqual(loss, 0.5 * lam * (self.w @ self.w), places=14)
        np.testing.assert_allclose(grad, lam * self.w, atol=1e-15)
        self.check_gradient(y, lam)

    def test_repeating_questions_preserves_mean(self):
        loss, grad = group_loss_grad(self.z, self.y, 0.13, self.w)
        repeated = group_loss_grad(np.tile(self.z, (3, 1)),
                                   np.tile(self.y, (3, 1)), 0.13, self.w)
        self.assertAlmostEqual(loss, repeated[0], places=14)
        np.testing.assert_allclose(grad, repeated[1], atol=1e-14)

    def test_extreme_scores_remain_finite(self):
        z = np.array([[1.0], [-1.0], [0.0], [0.5]])
        y = np.array([[0.0, 1.0, 0.0, 0.0]])
        loss, grad = group_loss_grad(z, y, 0.0, np.array([1000.0]))
        self.assertEqual(loss, 2000.0)
        np.testing.assert_allclose(grad, [2.0])

    def test_no_positive_is_rejected(self):
        y = self.y.copy()
        y[0] = 0
        with self.assertRaisesRegex(ValueError, 'at least one positive'):
            group_loss_grad(self.z, y, 0.0, self.w)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GroupGradientTest))
    if not result.wasSuccessful():
        sys.exit(1)
    t = GroupGradientTest()
    t.setUp()
    print(json.dumps({
        'synthetic_seed': 7506,
        'tests_passed': result.testsRun,
        'finite_difference_step': 1e-6,
        'max_abs_gradient_error_no_l2': t.check_gradient(t.y, 0.0),
        'max_abs_gradient_error_with_l2': t.check_gradient(t.y, 0.13),
        'max_abs_gradient_error_all_positive_with_l2': t.check_gradient(np.ones_like(t.y), 0.13),
    }, indent=2))
