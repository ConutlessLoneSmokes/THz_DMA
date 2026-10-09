"""One check that frequency-wise SOMP and stacked OMP use different scores."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thz_dma.estimators.planar_sparse import (  # noqa: E402
    PlanarTemplateDictionary,
    omp_three_path_estimate,
    somp_three_path_estimate,
)
from thz_dma.observations.planar import _build_schedule  # noqa: E402
from thz_dma.surfaces.multistrip import cyclic_frequency_resonance_states  # noqa: E402


class SompTest(unittest.TestCase):
    def test_frozen_c2_state_matches_cyclic_bank(self) -> None:
        state = {
            "name": "C2", "family": "frequency_single_shot",
            "num_configurations": 1, "pilots_per_configuration": 128,
            "state_design": "cyclic_frequency_state", "bank_size": 16, "state_index": 2,
        }
        config = {
            "system": {"num_subcarriers": 128, "carrier_frequency_hz": 3e11},
            "dma": {"num_microstrips": 8, "elements_per_microstrip": 160,
                    "tuning_bandwidth_hz": 3e10},
        }
        actual = _build_schedule(state, config).resonant_frequencies_hz
        expected = cyclic_frequency_resonance_states(16, 8, 160, 3e11, 3e10)[2:3]
        np.testing.assert_array_equal(actual, expected)

    def test_common_support_uses_per_tone_score(self) -> None:
        dictionary = PlanarTemplateDictionary(
            np.array([1.0, 2.0]),
            np.zeros(2),
            np.zeros(2),
            np.array([[1, -1, 0, 0], [1, 0, 0, 1]], dtype=complex),
        )
        observed = np.array([1, 1, 0, 0], dtype=complex)
        omp = omp_three_path_estimate(dictionary, observed, num_paths=1)
        somp = somp_three_path_estimate(
            dictionary, observed, observation_shape=(2, 1, 2), num_paths=1
        )
        self.assertEqual(omp["candidate_index"].tolist(), [1])
        self.assertEqual(somp["candidate_index"].tolist(), [0])
        self.assertAlmostEqual(somp["residual_energy"], 2.0)
        with self.assertRaises(ValueError):
            somp_three_path_estimate(
                dictionary, observed, observation_shape=(2, 1, 1), num_paths=1
            )


if __name__ == "__main__":
    unittest.main()
