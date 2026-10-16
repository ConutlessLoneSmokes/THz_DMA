"""Independent checks of sampled REs, state diversity and validation selection."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from _run_support import read_toml
from run_d041_configuration_sweep import plans, select_test_operators, summarize
from test_d041_diagnostic import config_for
from thz_dma.observations.pilots import full_band_pilot_indices
from zhou2025_repro.baselines import real_observation
from zhou2025_repro.configuration_sweep import ScheduledDmaOperator, re128_mask
from zhou2025_repro.d041 import SpectralDmaOperator, diagnostics
from zhou2025_repro.data import decode, encode


class ScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def operator(self, active=None):
        rng = np.random.default_rng(73)
        weights = rng.normal(size=(3, 5, 4))+1j*rng.normal(size=(3, 5, 4))
        if active is None:
            active = np.array([[1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [0, 0, 1, 1, 0]], dtype=bool)
        return weights, ScheduledDmaOperator(weights, 2, active, rf_noise_fraction=.2, dtype=torch.complex128)

    def test_compact_forward_adjoint_and_gaussian_gradient(self):
        weights, op = self.operator()
        image = torch.randn(2, *op.image_shape, dtype=torch.float64)
        observation = torch.randn(2, *op.observation_shape, dtype=torch.complex128)
        physical = decode(image, "dma")
        raw = np.einsum("jke,bkse->bsjk", weights.conj(), physical.numpy())
        direct = raw/np.sqrt(np.sum(abs(weights)**2, axis=-1)+.2)[None, None]
        torch.testing.assert_close(op.forward(image), torch.as_tensor(direct[..., op.active]), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close((op.forward(image).conj()*observation).sum().real,
            (image*op.adjoint(observation)).sum(), rtol=1e-12, atol=1e-12)
        size = int(np.prod(op.image_shape))
        basis = torch.eye(size, dtype=torch.float64).reshape(size, *op.image_shape)
        matrix = real_observation(op.forward(basis)).T
        for bar in (.03, .6, .9999):
            state = image.clone().requires_grad_(True)
            residual = real_observation(observation)-state.flatten(1)@matrix.T/math.sqrt(bar)
            covariance = (1-bar)/bar*(matrix@matrix.T)+.15*torch.eye(len(matrix), dtype=matrix.dtype)
            likelihood = -.5*(residual*torch.linalg.solve(covariance, residual.T).T).sum()
            expected, = torch.autograd.grad(likelihood, state)
            torch.testing.assert_close(op.likelihood_score(image, observation, .3, bar), expected, rtol=1e-9, atol=1e-9)

    def test_unobserved_frequencies_are_null_space(self):
        _, op = self.operator()
        truth = torch.randn(1, 5, 2, 4, dtype=torch.complex128)
        clean = op.forward(encode(truth, "dma"))
        estimate = op.least_squares(clean)
        torch.testing.assert_close(estimate, op.project_row(truth), rtol=1e-11, atol=1e-11)
        self.assertEqual(int(op.ranks[1]), 0)
        self.assertEqual(int(op.ranks[4]), 0)
        torch.testing.assert_close(estimate[:, [1, 4]], torch.zeros_like(estimate[:, [1, 4]]), rtol=0, atol=1e-12)
        null = truth-estimate
        torch.testing.assert_close(op.forward(encode(null, "dma")), torch.zeros_like(clean), rtol=0, atol=1e-11)
        self.assertLess(diagnostics(op, truth, estimate, clean, clean)["error_decomposition_relative_gap"], 1e-12)

    def test_all_re_mask_matches_original_operator(self):
        weights, sampled = self.operator(np.ones((3, 5), dtype=bool))
        original = SpectralDmaOperator(weights, 2, rf_noise_fraction=.2, dtype=torch.complex128)
        image = torch.randn(1, *original.image_shape, dtype=torch.float64)
        y = original.forward(image)
        torch.testing.assert_close(sampled.forward(image), y.flatten(2), rtol=0, atol=0)
        torch.testing.assert_close(sampled.likelihood_score(image, y.flatten(2), .1, .8),
            original.likelihood_score(image, y, .1, .8), rtol=1e-12, atol=1e-12)

    def test_repeated_state_does_not_add_noiseless_rank(self):
        weights, _ = self.operator()
        active = np.ones((3, 5), dtype=bool)
        diverse = ScheduledDmaOperator(weights, 2, active, dtype=torch.complex128)
        repeated = ScheduledDmaOperator(np.repeat(weights[:1], 3, axis=0), 2, active, dtype=torch.complex128)
        self.assertEqual(diverse.ranks.tolist(), [3]*5)
        self.assertEqual(repeated.ranks.tolist(), [1]*5)
        truth = torch.randn(1, 5, 2, 4, dtype=torch.complex128)
        single = SpectralDmaOperator(weights[:1], 2, dtype=torch.complex128)
        torch.testing.assert_close(repeated.project_row(truth), single.project_row(truth), rtol=1e-12, atol=1e-12)

    def test_resource_budget_and_frequency_coverage(self):
        for count in (1, 2, 4, 8, 16):
            repeated = re128_mask(count)
            interleaved = re128_mask(count, interleaved=True)
            self.assertEqual(int(repeated.sum()), 128)
            self.assertEqual(int(interleaved.sum()), 128)
            self.assertEqual(int(repeated.any(0).sum()), 128//count)
            self.assertEqual(int(interleaved.any(0).sum()), 128)
            np.testing.assert_array_equal(interleaved.sum(0), 1)
        with self.assertRaises(ValueError):
            re128_mask(3)
        with self.assertRaises(ValueError):
            self.operator(np.zeros((3, 5), dtype=bool))

    def test_f1_c2_and_f2_state_set_and_matched_repeat(self):
        config = config_for("wide")
        config["configuration_sweep"] = {**read_toml(ROOT/"configs/d041_configuration_sweep.toml"), "experiment": "re128"}
        schedules = list(plans(config))
        self.assertEqual(len(schedules), 13)
        self.assertEqual(schedules[0][3]["state_indices"], [2])
        f2 = next(p for p in schedules if p[0] == "re128_J8_K16_distinct")
        self.assertEqual(sorted(f2[3]["state_indices"]), list(range(0, 16, 2)))
        np.testing.assert_array_equal(np.flatnonzero(f2[2][0]), full_band_pilot_indices(128, 16))
        for count in (2, 4, 8, 16):
            a = next(p for p in schedules if p[0] == f"re128_J{count}_K{128//count}_distinct")
            b = next(p for p in schedules if p[0] == f"re128_J{count}_K{128//count}_repeat")
            np.testing.assert_array_equal(a[1][0], b[1][0])
            np.testing.assert_array_equal(b[1], np.repeat(a[1][:1], count, axis=0))
            np.testing.assert_array_equal(a[2], b[2])


class SelectionTest(unittest.TestCase):
    def config(self):
        config = config_for("narrow")
        config["job"]["smoke"] = False
        config["configuration_sweep"] = {**read_toml(ROOT/"configs/d041_configuration_sweep.toml"), "experiment": "curve"}
        return config

    def summary(self):
        return pd.DataFrame([{"operator": f"dma_J{j}", "state_mode": "distinct", "configurations": j,
            "recovery_scene_fraction": recovery, "near_reference_scene_fraction": fraction,
            "mean_nmse_loss_vs_J192_db": loss} for j, recovery, fraction, loss in
            [(192, 1., 1., 0.), (160, 1., 1., .1), (159, 1., .96, .4), (158, 1., .9, .4), (32, .95, 0., 40.), (8, 0., 0., 60.)]])

    def test_minima_selected_from_validation_with_both_near_reference_gates(self):
        selection = select_test_operators(self.summary(), self.config())
        self.assertEqual(selection["minimum_near_reference_J"], 159)
        self.assertEqual(selection["minimum_recovery_J"], 32)
        self.assertIn("dma_J158", selection["operator_labels"])
        self.assertNotIn("dma_J157", selection["operator_labels"])

    def test_smoke_never_declares_performance_threshold(self):
        config = self.config()
        config["job"]["smoke"] = True
        selection = select_test_operators(self.summary(), config)
        self.assertIsNone(selection["minimum_near_reference_J"])
        self.assertIsNone(selection["minimum_recovery_J"])

    def test_worst_variance_uses_linear_means_and_paired_scenes(self):
        rows, operators = [], []
        for label, count, error in (("dma_J192", 192, .001), ("dma_J8", 8, .01)):
            operators.append({"label": label, "configurations": count, "state_mode": "distinct",
                "pilot_re": count, "complex_observations": count*8, "rank": count*8, "switches": count-1})
            for scene in ("s0", "s1"):
                for variance, gain in (("v1", 1.), ("v2", 2.)):
                    rows.append({"operator": label, "condition": variance, "scene_id": scene,
                        "estimator": "clean_ddpm", "nmse_linear": error*gain, "nmse_db": 10*math.log10(error*gain),
                        "observation_residual_relative": 0., "prediction_energy_ratio": 1., "normalized_correlation": .9,
                        "row_space_nmse_linear": 0., "null_space_nmse_linear": error*gain})
        summary = summarize(pd.DataFrame(rows), operators, self.config())
        self.assertAlmostEqual(summary.iloc[1].mean_nmse_loss_vs_J192_db, 10.)
        self.assertAlmostEqual(summary.iloc[1].worst_variance_mean_nmse_db, 10*math.log10(.02))
        self.assertFalse(summary.iloc[0].numerical_variance_stable)


if __name__ == "__main__":
    unittest.main()
