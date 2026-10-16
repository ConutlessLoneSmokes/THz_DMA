"""Independent algebra, physical pairing, variance and trajectory controls."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from run_d041_diagnostic import resolved_settings, training_reference, sample_observation, noise_like, posterior_seed
from zhou2025_repro.baselines import real_observation
from zhou2025_repro.data import encode, decode
from zhou2025_repro.ddpm import DenoisingCNN, VPSchedule, recover
from zhou2025_repro.d041 import (
    DiagnosticDigitalOperator, SpectralDmaOperator, diagnostics, geometry_ls,
    geometry_templates, hardware_weights, paired_array, paired_channel, state_bank,
)


def config_for(band):
    return resolved_settings(SimpleNamespace(config=ROOT/"configs/zhou2025_d041.toml", band=band,
        device="cpu", smoke=True, command="prepare", run_id="fixture", data_run_id=None,
        train_run_id=None, stage="full-noiseless", split="validation", tuning_run_id=None, resume=False))


class DiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_array_fft_layout_roundtrip_and_energy(self):
        torch.manual_seed(59)
        for dtype in (torch.complex64, torch.complex128):
            physical = torch.randn(1, 128, 8, 160, dtype=dtype)
            image = encode(physical, "dma")
            torch.testing.assert_close(decode(image, "dma"), physical)
            torch.testing.assert_close(image.square().sum(), physical.abs().square().sum())

    def test_paired_physical_paths_and_true_single_frequency(self):
        narrow, wide = config_for("narrow"), config_for("wide")
        a, b = paired_array(narrow), paired_array(wide)
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.feed_distances, b.feed_distances)
        self.assertEqual(a.frequencies.tolist(), [3e11])
        self.assertEqual(len(b.frequencies), 128)
        ha, ra = paired_channel(narrow, a, 11, 2)
        hb, rb = paired_channel(wide, b, 11, 2)
        self.assertEqual(ra, rb)
        self.assertEqual(ha.shape, (1, 8, 160))
        self.assertEqual(hb.shape, (128, 8, 160))
        # The original even OFDM grid excludes fc: compare against a genuine fc evaluation.
        central = copy.copy(b)
        object.__setattr__(central, "frequencies", a.frequencies)
        hc, rc = paired_channel(wide, central, 11, 2)
        np.testing.assert_array_equal(ha, hc)
        self.assertEqual(ra, rc)
        gains = np.asarray(ra["gain_real"])+1j*np.asarray(ra["gain_imag"])
        np.testing.assert_allclose(np.einsum("l,lkse->kse", gains, geometry_templates(a, ra)), ha)
        narrow["channel"]["include_absorption"] = True
        with self.assertRaises(ValueError):
            paired_array(narrow)

    def test_state_bank_independent_of_band_and_channel_seeds(self):
        narrow, wide = config_for("narrow"), config_for("wide")
        np.testing.assert_array_equal(state_bank(narrow), state_bank(wide))
        narrow["data"]["test_seed"] += 100
        np.testing.assert_array_equal(state_bank(narrow), state_bank(wide))
        bank = state_bank(narrow)
        self.assertEqual(bank.shape, (192, 1, 160))
        self.assertGreaterEqual(bank.min(), 285e9)
        self.assertLessEqual(bank.max(), 315e9)
        self.assertEqual(hardware_weights(narrow, paired_array(narrow), bank).shape, (192, 1, 160))

    def test_spectral_adjoint_and_real_likelihood_gradient(self):
        rng = np.random.default_rng(52)
        for configurations in (2, 6):
            weights = rng.normal(size=(configurations, 2, 4))+1j*rng.normal(size=(configurations, 2, 4))
            op = SpectralDmaOperator(weights, 2, rf_noise_fraction=.2, dtype=torch.complex128)
            size = int(np.prod(op.image_shape))
            basis = torch.eye(size, dtype=torch.float64).reshape(size, *op.image_shape)
            matrix = real_observation(op.forward(basis)).T
            torch.manual_seed(54)
            image = torch.randn(2, *op.image_shape, dtype=torch.float64)
            observation = torch.randn_like(op.forward(image))
            torch.testing.assert_close((op.forward(image).conj()*observation).sum().real,
                (image*op.adjoint(observation)).sum(), rtol=1e-12, atol=1e-12)
            for bar in (.03, .6, .9999):
                state = image.clone().requires_grad_(True)
                residual = real_observation(observation)-state.flatten(1)@matrix.T/math_sqrt(bar)
                covariance = (1-bar)/bar*(matrix@matrix.T)+.15*torch.eye(len(matrix), dtype=matrix.dtype)
                likelihood = -.5*(residual*torch.linalg.solve(covariance, residual.T).T).sum()
                expected, = torch.autograd.grad(likelihood, state)
                torch.testing.assert_close(op.likelihood_score(image, observation, .3, bar), expected,
                    rtol=1e-9, atol=1e-9)

    def test_pseudoinverse_row_null_decomposition_and_genie_gains(self):
        rng = np.random.default_rng(55)
        for configurations in (2, 7):
            weights = rng.normal(size=(configurations, 3, 4))+1j*rng.normal(size=(configurations, 3, 4))
            op = SpectralDmaOperator(weights, 2, dtype=torch.complex128)
            physical = torch.randn(1, 3, 2, 4, dtype=torch.complex128)
            y = op.forward(encode(physical, "dma"))
            estimate = op.least_squares(y)
            row = op.project_row(physical)
            torch.testing.assert_close(estimate, row, rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(op.project_row(row), row, rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(op.forward(encode(physical-row, "dma")), torch.zeros_like(y),
                rtol=0, atol=1e-11)
            if configurations >= 4:
                torch.testing.assert_close(physical, estimate, rtol=1e-11, atol=1e-11)
            d = diagnostics(op, physical, estimate, y, y)
            self.assertLess(d["error_decomposition_relative_gap"], 1e-12)
            self.assertLess(d["row_space_nmse_linear"], 1e-24)
            templates = torch.randn(3, 3, 2, 4, dtype=torch.complex128)
            gains = torch.randn(3, dtype=torch.complex128)
            truth = torch.einsum("l,lkse->kse", gains, templates)[None]
            observed = op.forward(encode(truth, "dma"))
            torch.testing.assert_close(geometry_ls(op, observed, templates), truth, rtol=1e-11, atol=1e-11)

    def test_trace_observer_does_not_change_frozen_endpoint(self):
        torch.manual_seed(56)
        op = DiagnosticDigitalOperator((1, 2, 4))
        model, schedule = DenoisingCNN(channels=4, width=8), VPSchedule(8, 1e-4, .3)
        y = torch.randn(1, *op.observation_shape, dtype=torch.complex64)
        expected, trace = recover(model, schedule, op, y, 1e-8, seed=57)
        captured = []
        def observer(state, info):
            captured.append((state.clone(), info))
        actual, actual_trace = recover(model, schedule, op, y, 1e-8, seed=57, trace_observer=observer)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(trace, actual_trace)
        self.assertEqual(len(captured), 9)
        self.assertEqual(captured[0][1]["phase"], "initial")
        self.assertEqual(captured[-1][1]["t"], 0)
        self.assertEqual(captured[-1][1]["state_alpha_bar"], 1.)

    def test_training_only_snr_calibration_and_noiseless_variance(self):
        config = config_for("narrow")
        op = DiagnosticDigitalOperator((1, 1, 1))
        class FakeData:
            counts = {"train": 2}
            scale = 3.
            evaluation_gain = 1.
            def images(self, split, indices, device):
                self.assert_training(split)
                return encode(torch.ones(len(indices), 1, 1, 1, dtype=torch.complex64)*2, "dma")
            def assert_training(self, split):
                if split != "train":
                    raise AssertionError("SNR calibration read non-training truth")
            def physical(self, split, indices):
                return np.ones((len(indices), 1, 1, 1), dtype=np.complex64)*self.evaluation_gain
        data = FakeData()
        reference = training_reference(data, op, config)
        self.assertEqual(reference, 4.)
        data.evaluation_gain = 100.
        self.assertEqual(training_reference(data, op, config), reference)
        truth, clean, observed = sample_observation(data, op, config, "digital", 0, "validation", 1e-8, True)
        torch.testing.assert_close(clean, observed, rtol=0, atol=0)
        self.assertAlmostEqual(float(truth.abs().mean()), 100./3, places=4)

    def test_configuration_curve_pairs_initialization_and_standard_noise(self):
        config = config_for("wide")
        seed = posterior_seed(config, "digital", 7, "test")
        for label in ("dma_J192", "dma_J32", "dma_J8", "dma_J1"):
            self.assertEqual(posterior_seed(config, label, 7, "test"), seed)
        full = noise_like(torch.zeros(1, 8, 192, 128, dtype=torch.complex64), config, "dma_J192", 7, "test")
        for count in (32, 8, 1):
            reduced = noise_like(torch.zeros(1, 8, count, 128, dtype=torch.complex64), config, f"dma_J{count}", 7, "test")
            torch.testing.assert_close(full[:, :, :count], reduced, rtol=0, atol=0)


def math_sqrt(value):
    return value**.5


if __name__ == "__main__":
    unittest.main()
