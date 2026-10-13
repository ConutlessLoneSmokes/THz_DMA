"""Small closed-form checks for the local MIMO traditional references."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from _run_support import read_toml
from zhou2025_repro.data import angular, encode
from zhou2025_repro.mimo import MimoOperator, parametric_mimo
from zhou2025_repro.parametric_baselines import (angular_omp, mimo_path_templates,
                                                oracle_true_geometry)


class ParametricBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        config = read_toml(ROOT / "configs/zhou2025_parametric_mimo.toml")
        config["mimo"].update(receive_antennas=4, transmit_antennas=8)
        config["channel"].update(num_paths=2, include_absorption=False)
        return config

    def test_angular_omp_sparse_recovery_and_zero(self):
        torch.manual_seed(10)
        b = torch.fft.fft(torch.eye(8, dtype=torch.complex128), norm="ortho")
        channel = torch.zeros(2, 3, 8, dtype=b.dtype)
        channel[..., 2] = torch.randn(2, 3, dtype=b.dtype)
        channel[..., 6] = torch.randn(2, 3, dtype=b.dtype)
        estimate, diagnostic = angular_omp(channel @ b, b, max_atoms=2)
        torch.testing.assert_close(estimate, channel, rtol=1e-12, atol=1e-12)
        self.assertLess(diagnostic["residual_energy"].max(), 1e-24)
        self.assertTrue(np.all(np.sort(diagnostic["candidate_index"], axis=-1) == [2, 6]))
        zero, _ = angular_omp(torch.zeros(1, 1, 8, dtype=b.dtype), b)
        torch.testing.assert_close(zero, torch.zeros_like(zero))

    def test_unit_gain_templates_and_oracle_gain_fitting(self):
        config = self.config()
        channel, paths = parametric_mimo(config, np.random.default_rng(17), None)
        templates = mimo_path_templates(config, paths, None)
        gains = np.array([complex(p["gain_real"], p["gain_imag"]) for p in paths])
        np.testing.assert_allclose(np.einsum("l,lrt->rt", gains, templates),
                                   channel, rtol=2e-7, atol=1e-10)
        hidden = copy.deepcopy(paths)
        for path in hidden:
            path["gain_real"], path["gain_imag"] = float("nan"), float("nan")
        np.testing.assert_array_equal(mimo_path_templates(config, hidden, None), templates)
        op = MimoOperator(torch.eye(8, dtype=torch.complex128), 4)
        scale = .017
        physical = torch.as_tensor(channel[None], dtype=torch.complex128) / scale
        y = op.forward(encode(physical, "mimo"))
        estimate, diagnostic = oracle_true_geometry(y, op.b, templates / scale)
        torch.testing.assert_close(estimate, angular(physical), rtol=2e-7, atol=1e-10)
        np.testing.assert_allclose(diagnostic["complex_gain"][0], gains, rtol=2e-7, atol=1e-8)
        self.assertTrue(diagnostic["oracle_geometry"])
        # The oracle must refit unknown gains from y, not replay saved truth.
        scaled, scaled_diagnostic = oracle_true_geometry(2 * y, op.b, templates / scale)
        torch.testing.assert_close(scaled, 2 * estimate)
        np.testing.assert_allclose(scaled_diagnostic["complex_gain"],
                                   2 * diagnostic["complex_gain"], rtol=1e-12, atol=1e-12)

    def test_spacing_priority_and_absorption(self):
        config = self.config()
        _, paths = parametric_mimo(config, np.random.default_rng(18), None)
        original = mimo_path_templates(config, paths, None)
        from thz_dma.channels.atmosphere import C0_M_PER_S
        config["mimo"]["spacing_m"] = .5 * C0_M_PER_S / config["system"]["carrier_frequency_hz"]
        config["mimo"]["spacing_wavelengths"] = 1.4
        np.testing.assert_array_equal(mimo_path_templates(config, paths, None), original)
        class ConstantAbsorption:
            def amplitude_transmission(self, frequency, distance, method):
                return .7
        config["channel"]["include_absorption"] = True
        np.testing.assert_allclose(mimo_path_templates(config, paths, ConstantAbsorption()),
                                   .7 * original, rtol=1e-14, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "absorption table"):
            mimo_path_templates(config, paths, None)

    def test_invalid_inputs(self):
        b = torch.eye(3, dtype=torch.complex128)
        y = torch.ones(1, 2, 3, dtype=b.dtype)
        for atoms in (0, 4, 1.5, True):
            with self.assertRaises(ValueError):
                angular_omp(y, b, atoms)
        with self.assertRaises(ValueError):
            angular_omp(y[..., :2], b)
        with self.assertRaises(ValueError):
            angular_omp(y * float("nan"), b)
        with self.assertRaisesRegex(ValueError, "zero-energy"):
            angular_omp(y, torch.zeros_like(b))
        with self.assertRaises(ValueError):
            oracle_true_geometry(y, b, np.ones((1, 3, 3), dtype=complex))
        with self.assertRaisesRegex(ValueError, "zero-energy"):
            oracle_true_geometry(y, b, np.zeros((1, 2, 3), dtype=complex))
        config = self.config()
        with self.assertRaises(ValueError):
            mimo_path_templates(config, [], None)
        config["system"]["carrier_frequency_hz"] = 0.
        with self.assertRaises(ValueError):
            mimo_path_templates(config, [], None)


if __name__ == "__main__":
    unittest.main()
