"""Small checks for helpers shared by the refactored stage runners."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thz_dma.estimators.los_grid import regular_grid  # noqa: E402
from thz_dma.evaluation.paths import match_two_paths  # noqa: E402
from thz_dma.metrics.diagnostics import channel_domain_metrics  # noqa: E402
from thz_dma.observations.pilots import full_band_pilot_indices  # noqa: E402


class SharedHelpersTest(unittest.TestCase):
    def test_grid_pilots_and_path_matching(self) -> None:
        np.testing.assert_array_equal(regular_grid(0.0, 1.0, 0.25), np.linspace(0, 1, 5))
        np.testing.assert_array_equal(full_band_pilot_indices(8, 1), [4])
        self.assertEqual(np.unique(full_band_pilot_indices(128, 16)).size, 16)

        ranges, angles, gains = match_two_paths(
            np.array([2.0, 1.0]),
            np.array([20.0, 10.0]),
            np.array([2.0j, 1.0j]),
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),
            range_scale_m=1.0,
            angle_scale_deg=1.0,
        )
        np.testing.assert_array_equal(ranges, [1.0, 2.0])
        np.testing.assert_array_equal(angles, [10.0, 20.0])
        np.testing.assert_array_equal(gains, [1.0j, 2.0j])

    def test_channel_domains_keep_empty_heldout_as_missing(self) -> None:
        truth = np.ones((128, 2), dtype=complex)
        metrics = channel_domain_metrics(truth, truth, np.arange(128), np.arange(16))
        self.assertEqual(metrics["heldout_112_nmse_subcarrier_count"], 0)
        self.assertIsNone(metrics["heldout_112_nmse_linear"])
        self.assertEqual(metrics["fullband_128_nmse_linear"], 0.0)


if __name__ == "__main__":
    unittest.main()
