"""Gaussian algebra, proper-complex factors, frozen recursion and validation gates."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from _run_support import RunArchive, read_toml
import run_d041_fusion_probe as probe
from test_d041_diagnostic import config_for
from zhou2025_repro.baselines import real_observation
from zhou2025_repro.configuration_sweep import ScheduledDmaOperator
from zhou2025_repro.d041 import SpectralDmaOperator
from zhou2025_repro.data import decode, encode
from zhou2025_repro.ddpm import VPSchedule, recover
from zhou2025_repro.moment_recovery import clean_prediction, condition_gaussian, fuse_scalar, gaussian_control, recover_moments


class FixedDenoiser(torch.nn.Module):
    def forward(self, state, time_index):
        return .1*state


class FusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def operator(self, scheduled=True, configurations=2):
        rng = np.random.default_rng(20261016)
        weights = rng.normal(size=(configurations, 3, 4))+1j*rng.normal(size=(configurations, 3, 4))
        if scheduled:
            active = np.tile([True, False, True], (configurations, 1))
            return ScheduledDmaOperator(weights, 2, active, dtype=torch.complex128)
        return SpectralDmaOperator(weights, 2, dtype=torch.complex128)

    def test_general_gaussian_local_mean_and_covariance(self):
        result = gaussian_control()
        self.assertTrue(result["passed"])
        self.assertLess(max(result["relative_errors"]), 1e-10)

    def test_scalar_svd_matches_dense_real_conditional_including_factor_two(self):
        for scheduled in (False, True):
            for configurations in (2, 6):
                op = self.operator(scheduled, configurations)
                size = int(np.prod(op.image_shape))
                basis = torch.eye(size, dtype=torch.float64).reshape(size, *op.image_shape)
                matrix = real_observation(op.forward(basis)).T
                mean = torch.randn(1, *op.image_shape, dtype=torch.float64)
                observed = torch.randn(1, *op.observation_shape, dtype=torch.complex128)
                nu, v = .13, .07
                expected, _ = condition_gaussian(mean.flatten(), v*torch.eye(size, dtype=torch.float64),
                    matrix, real_observation(observed).flatten(), .5*nu*torch.eye(len(matrix), dtype=torch.float64))
                actual = fuse_scalar(mean, op, observed, nu, v)
                torch.testing.assert_close(actual.flatten(), expected, rtol=1e-11, atol=1e-11)

    def test_scalar_fusion_changes_only_row_space(self):
        op = self.operator()
        mean = torch.randn(1, *op.image_shape, dtype=torch.float64)
        observed = torch.randn(1, *op.observation_shape, dtype=torch.complex128)
        fused = fuse_scalar(mean, op, observed, 1e-10, .3)
        correction = decode(fused-mean, "dma")
        torch.testing.assert_close(op.project_row(correction), correction, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(decode(fused, "dma")[:, 1], decode(mean, "dma")[:, 1], rtol=0, atol=1e-12)
        torch.testing.assert_close(fuse_scalar(mean, op, op.forward(mean), .2, .3), mean, rtol=0, atol=0)

    def test_exact_noise_prediction_recovers_clean(self):
        clean, noise = torch.randn(2, 2, 3, 4, dtype=torch.float64), torch.randn(2, 2, 3, 4, dtype=torch.float64)
        for bar in (.01, .6, .9999):
            state = math.sqrt(bar)*clean+math.sqrt(1-bar)*noise
            torch.testing.assert_close(clean_prediction(state, noise, bar), clean, rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            clean_prediction(clean, noise, 1.)

    def test_fusion_input_validation(self):
        op = self.operator()
        mean = torch.randn(1, *op.image_shape, dtype=torch.float64)
        observed = op.forward(mean)
        for noise, v in [(0, .1), (.1, -1), (float("nan"), .1), (.1, float("inf"))]:
            with self.assertRaises(ValueError):
                fuse_scalar(mean, op, observed, noise, v)
        with self.assertRaises(ValueError):
            fuse_scalar(mean, op, observed.repeat(2, 1, 1), .1, .1)

    def test_dense_covariance_validation(self):
        mean, matrix, observed = torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            condition_gaussian(mean, matrix, matrix, observed, -matrix)
        with self.assertRaises(ValueError):
            condition_gaussian(mean, matrix, matrix[:, :1], observed, matrix)
        with self.assertRaises(torch.linalg.LinAlgError):
            condition_gaussian(mean, -matrix, matrix, observed, matrix)

    def test_no_fusion_reduces_to_original_unguided_recursion(self):
        op, schedule, model = self.operator(), VPSchedule(8, 1e-4, .3), FixedDenoiser()
        observed = torch.randn(1, *op.observation_shape, dtype=torch.complex128)
        captured = []
        with patch("zhou2025_repro.moment_recovery.fuse_scalar", side_effect=lambda mean, *args: mean):
            actual = recover_moments(model, schedule, op, observed, .1, variances_real=[.2]*8,
                seed=73, trace_observer=lambda state, info: captured.append((state.clone(), info)))
        expected, _ = recover(model, schedule, op, observed, .1, scale=0., seed=73)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(captured), 9)
        self.assertEqual(captured[-1][1]["network_calls"], 8)
        self.assertEqual(captured[-1][1]["state_alpha_bar"], 1.)

    def test_paired_initial_state_and_final_fused_mean(self):
        op, schedule, model = self.operator(), VPSchedule(8, 1e-4, .3), FixedDenoiser()
        observed = torch.randn(1, *op.observation_shape, dtype=torch.complex128)
        original, fused, means = [], [], []
        def callback(state, info):
            fused.append((state.clone(), info))
        wrapper = probe.ObservedDenoiser(model, schedule, lambda mean, info: means.append(mean.clone()), {1, 8})
        actual = recover_moments(wrapper, schedule, op, observed, .1, variances_real=[.2]*8,
            seed=79, trace_observer=callback)
        recover(model, schedule, op, observed, .1, seed=79, trace_observer=lambda state, info: original.append(state.clone()))
        torch.testing.assert_close(original[0], fused[0][0], rtol=0, atol=0)
        torch.testing.assert_close(actual, fuse_scalar(means[-1], op, observed, .1, .2), rtol=1e-11, atol=1e-11)
        self.assertEqual(wrapper.calls, 8)
        with self.assertRaises(ValueError):
            recover_moments(model, schedule, op, observed, .1, variances_real=[.2]*7)

    def test_compact_noise_gathers_common_grid_including_more_than192_re(self):
        config = config_for("wide")
        weights = np.ones((3, 128, 2), dtype=np.complex128)
        active = np.ones((3, 128), dtype=bool)
        full = ScheduledDmaOperator(weights, 2, active, dtype=torch.complex128)
        subset = active.copy()
        subset[:, 1::2] = False
        partial = ScheduledDmaOperator(weights, 2, subset, dtype=torch.complex128)
        standard = probe.common_noise(full, config, 2)
        self.assertEqual(standard.shape, (1, 2, 384))
        selected = torch.as_tensor(standard.numpy().reshape(1, 2, 3, 128)[:, :, subset])
        torch.testing.assert_close(probe.common_noise(partial, config, 2), selected, rtol=0, atol=0)
        self.assertFalse(torch.equal(standard, probe.common_noise(full, config, 3)))

    def test_predeclared_protocol_and_disjoint_validation(self):
        args = SimpleNamespace(config=ROOT/"configs/d041_fusion_probe.toml", data_run_id="fixture_data",
            train_run_id="fixture_model", run_id="fixture", smoke=False, device="cpu")
        source = config_for("wide")
        with patch.object(probe, "read_json", return_value={"config": source}):
            config = probe.resolved_settings(args)
            settings = config["fusion_probe"]
            self.assertEqual(sum(len(probe.conditions(settings, r)) for r in settings["operators"]), 12)
            self.assertGreaterEqual(settings["screening_start"], settings["tuning_scenes"])
            invalid = copy.deepcopy(settings)
            invalid["screening_start"] = 0
            with patch.object(probe, "read_toml", return_value=invalid), self.assertRaises(ValueError):
                probe.resolved_settings(args)
            args.smoke = True
            small = probe.resolved_settings(args)
            self.assertEqual(small["diffusion"], source["diffusion"])
            self.assertEqual(small["system"], source["system"])

    def test_selection_excludes_nonfinite_and_reports_boundary(self):
        rows = [{"method": "zhou", "guidance": g, "scene_index": s, "status": "success", "nmse_linear": e}
                for g, e in [(4., .2), (8., .1), (16., .05)] for s in (0, 1)]
        best = probe.select_guidance(rows, [4., 8., 16.])
        self.assertEqual(best["selected"], 16.)
        self.assertTrue(best["at_upper_boundary"])
        rows[-1].update(status="nonfinite", nmse_linear=None)
        self.assertEqual(probe.select_guidance(rows, [4., 8., 16.])["selected"], 8.)
        with self.assertRaises(ArithmeticError):
            probe.select_guidance(rows[-1:], [16.])

    def test_worst_variance_pairs_scenes_then_averages_linear_errors(self):
        rows = [{"operator": "a", "method": "zhou", "condition": c, "guidance": 4.,
                 "scene_index": s, "status": "success", "nmse_linear": e}
                for c, s, e in [("noiseless_a", 0, .01), ("noiseless_b", 0, .02),
                                ("noiseless_a", 1, .2), ("noiseless_b", 1, .1)]]
        tuning = {f"a:{c}": {"selected": 4.} for c in ("noiseless_a", "noiseless_b")}
        result = probe.worst_variance_summary(pd.DataFrame(rows), tuning)[0]
        self.assertAlmostEqual(result["worst_variance_mean_nmse_db"], 10*math.log10(.11))
        self.assertEqual(result["fraction_below_minus10db"], .5)
        self.assertFalse(result["numerical_variance_stable"])
        rows[-1]["status"] = "nonfinite"
        result = probe.worst_variance_summary(pd.DataFrame(rows), tuning)[0]
        self.assertEqual(result["failed_scenes"], 1)
        self.assertIsNone(result["worst_variance_mean_nmse_db"])
        self.assertEqual(result["fraction_below_minus10db"], .5)

    def test_variance_calibration_uses_training_only_and_never_reads_test(self):
        op, schedule = self.operator(), VPSchedule(2, 1e-4, .2)
        config = config_for("wide")
        config["fusion_probe"] = {**read_toml(ROOT/"configs/d041_fusion_probe.toml"),
            "calibration_train_scenes": 2, "prior_validation_scenes": 1, "trace_times": [1, 2]}
        accessed = []
        class Data:
            def images(self, split, indices, device):
                accessed.append(split)
                if split not in ("train", "validation"):
                    raise AssertionError("test access forbidden")
                return torch.full((len(indices), *op.image_shape), 1. if split == "train" else 100., dtype=torch.float64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/"checkpoint").mkdir()
            archive = RunArchive("fixture", path)
            with patch.object(probe, "operators", return_value=iter([("fixture", op, None)])):
                probe.calibrate(archive, config, FixedDenoiser(), schedule, Data())
            profile = json.loads((path/"calibration_profile.json").read_text())
            table = pd.read_csv(path/"prior_denoising.csv")
            for t, value in enumerate(profile["variances_real"], 1):
                block = table.loc[table.t == t]
                expected = block.loc[block.split == "train", "real_coordinate_mse"].mean()
                self.assertAlmostEqual(value, expected)
                self.assertNotAlmostEqual(value, block.real_coordinate_mse.mean())
            self.assertNotIn("test", accessed)


if __name__ == "__main__":
    unittest.main()
