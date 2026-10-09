"""Compare matrix-free linear estimators with dense reference equations."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thz_dma.estimators.linear_channel import lmmse_coefficients, ridge_channel


class LinearChannelTest(unittest.TestCase):
    def test_dense_covariance_and_tikhonov(self):
        rng = np.random.default_rng(42)
        draw = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
        h = draw((5, 7))
        a = draw((3, 7))
        y = draw((3, 2))
        variance = np.array([0.1, 2.0])
        b = h @ a.T
        actual = h.T @ lmmse_coefficients(b, y, variance)
        covariance = h.T @ h.conj()
        expected = np.column_stack([
            covariance @ a.conj().T @ np.linalg.solve(
                a @ covariance @ a.conj().T + v * np.eye(3), y[:, i])
            for i, v in enumerate(variance)
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        operator = SimpleNamespace(weights=a.conj()[:, None, None, :],
                                   relative_output_noise_variance=np.ones((1, 3, 1)),
                                   observation_shape=(1, 3, 1))
        actual = ridge_channel(operator, y[:, 0], 0.1).ravel()
        expected = np.linalg.solve(a.conj().T @ a + 0.1 * np.eye(7), a.conj().T @ y[:, 0])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        with self.assertRaises(ValueError):
            lmmse_coefficients(b, y, -1)


if __name__ == "__main__":
    unittest.main()
