"""SBL evidence updates, complex posterior and phase/noise convention checks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thz_dma.estimators.sbl import sparse_bayesian_estimate
from thz_dma.stage2a import _method_names


class SblTest(unittest.TestCase):
    def test_retired_estimators_are_not_scheduled(self):
        for retired in ("grid_somp_3d", "ridge"):
            with self.assertRaises(ValueError):
                _method_names({"experiment": {"estimators": ["oracle_geometry_ls", "grid_omp_3d", retired]}})

    def test_evidence_and_posterior(self):
        rng = np.random.default_rng(23)
        rows = rng.normal(size=(16, 24)) + 1j*rng.normal(size=(16, 24))
        y = rows[[2, 9]].T @ np.array([1+.4j, -.8j])
        noise = 0.02
        fit = sparse_bayesian_estimate(rows, y, noise)
        self.assertTrue(fit["converged"])
        self.assertEqual(set(fit["indices"]), {2, 9})
        self.assertTrue(np.all(np.diff(fit["objective_history"]) <= 1e-8))
        a = rows[fit["indices"]].T
        covariance = np.linalg.inv(a.conj().T @ a / noise + np.diag(1/fit["gamma"]))
        np.testing.assert_allclose(fit["mean"], covariance @ a.conj().T @ y / noise, rtol=1e-9)
        np.testing.assert_allclose(fit["variance"], covariance.diagonal().real, rtol=1e-9)
        c = noise*np.eye(len(y)) + (a*fit["gamma"]) @ a.conj().T
        expected = np.linalg.slogdet(c)[1] + np.vdot(y, np.linalg.solve(c, y)).real
        # Internal objective uses a rescaled observation and noise.
        self.assertAlmostEqual(fit["objective_history"][-1] + len(y)*np.log(np.vdot(y,y).real), expected, places=7)
        zero = sparse_bayesian_estimate(rows, np.zeros_like(y), noise)
        self.assertEqual(len(zero["indices"]), 0)
        with self.assertRaises(ValueError):
            sparse_bayesian_estimate(rows, y, 0)


if __name__ == "__main__":
    unittest.main()
