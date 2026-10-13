"""Physical/real gradients, VP scaling and SURE risk checks against closed forms."""
import sys
import unittest
import tempfile
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from thz_dma.channels.geometry import ula_positions, spherical_distances
from thz_dma.channels.planar import planar_dma_positions, planar_distances
from thz_dma.estimators.planar_sparse import PlanarDmaTemplateOperator
from zhou2025_repro.baselines import EmpiricalLMMSE, real_observation
from zhou2025_repro.data import encode, decode, signal_scale
from zhou2025_repro.ddpm import DenoisingCNN, VPSchedule, recover
from zhou2025_repro.dma import DmaOperator
from zhou2025_repro.mimo import MimoOperator
from zhou2025_repro.sure import risk, denoise, matched_time
import run_zhou2025_repro as runner
from _run_support import RunArchive
from zhou2025_repro.data import ChannelData


class ReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_unitary_geometry_and_data_order(self):
        line = ula_positions(7, .002)
        plane = planar_dma_positions(1, 7, .002, .003)
        np.testing.assert_array_equal(plane[0, :, 1], line)
        np.testing.assert_allclose(planar_distances(plane, 2., .2, 0.)[0],
                                   spherical_distances(line, 2., .2), atol=1e-14)
        torch.manual_seed(2)
        for scenario, shape in [("mimo", (3, 2, 8)), ("dma", (3, 4, 2, 8))]:
            channel = torch.randn(shape, dtype=torch.complex128)
            image = encode(channel, scenario)
            torch.testing.assert_close(decode(image, scenario), channel, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(image.square().sum(), channel.abs().square().sum())
        pilots = torch.randn(3, 2, dtype=torch.complex128)
        op = MimoOperator(pilots, 2)
        physical = torch.randn(2, 2, 3, dtype=torch.complex128)
        expected = torch.fft.ifft(physical @ pilots, dim=-2, norm="ortho")
        torch.testing.assert_close(op.forward(encode(physical, "mimo")), expected)
        image = encode(physical, "mimo")
        angle = torch.complex(image[:, 0], image[:, 1])
        column_matrix = torch.kron(op.b.T.contiguous(), torch.eye(2, dtype=pilots.dtype))
        vector = angle.transpose(-1, -2).flatten(1)
        received = op.forward(image).transpose(-1, -2).flatten(1)
        torch.testing.assert_close(vector@column_matrix.T, received)
        data = SimpleNamespace(scenario="dma", shape=(128,2,3), scale=1.,
                               record=lambda split,index: {"scene_id":"domain-check"})
        acquisition = SimpleNamespace(pilot_indices=np.arange(128), pilot_resource_elements=128,
                                      training_symbols=1, num_switches=0)
        metrics = runner.metric_row(data, np.ones(data.shape), np.zeros(data.shape), 0, "validation",
            "F1", "zero-control", 0., 1., torch.ones(1,2,2), acquisition,
            {"job":{"smoke":True}, "evaluation":{"channel_nmse_subcarrier_count":16},
             "evaluation_run":{}}, (0,None), 0.)
        self.assertEqual(metrics["channel_nmse_subcarrier_count"], 16)
        self.assertEqual(metrics["fullband_128_nmse_subcarrier_count"], 128)
        self.assertIsNone(metrics["heldout_112_nmse_linear"])

    def test_linear_likelihood_gradients(self):
        torch.manual_seed(3)
        pilots = torch.randn(3, 2, dtype=torch.complex128)
        weights = np.random.default_rng(3).normal(size=(2, 2, 2, 4)).astype(complex)
        weights += 1j * np.random.default_rng(4).normal(size=weights.shape)
        relative = np.sum(abs(weights)**2, axis=-1).transpose(2, 0, 1) + .2
        physical = PlanarDmaTemplateOperator(np.ones(2), np.zeros((2, 4, 3)),
                                            weights, np.zeros(2), relative)
        for op in (MimoOperator(pilots, 2),
                   DmaOperator(physical, [0, 3], tones=4, dtype=torch.complex128)):
            n = int(np.prod(op.image_shape))
            basis = torch.eye(n, dtype=torch.float64).reshape(n, *op.image_shape)
            matrix = real_observation(op.forward(basis)).T
            x = torch.randn(2, *op.image_shape, dtype=torch.float64)
            y = torch.randn_like(op.forward(x))
            torch.testing.assert_close((op.forward(x).conj()*y).sum().real,
                                       (x*op.adjoint(y)).sum())
            yr = real_observation(y)
            for bar in (.07, .6, .99):
                z = x.clone().requires_grad_(True)
                residual = yr - z.flatten(1) @ matrix.T / np.sqrt(bar)
                covariance = ((1-bar)/bar*(matrix@matrix.T)
                              + .15*torch.eye(len(matrix), dtype=matrix.dtype))
                log_likelihood = -.5*(residual*torch.linalg.solve(covariance, residual.T).T).sum()
                gradient, = torch.autograd.grad(log_likelihood, z)
                torch.testing.assert_close(op.likelihood_score(x, y, .3, bar), gradient,
                                           rtol=1e-9, atol=1e-9)

    def test_vp_noise_variance_and_frozen_endpoint(self):
        torch.manual_seed(4)
        schedule = VPSchedule(8, 1e-4, .2)
        clean = torch.ones(20000, 2, 1, 1, dtype=torch.float64) * 1.7
        indices = torch.full((len(clean),), 5, dtype=torch.long)
        noisy = schedule.perturb(clean, indices, torch.randn_like(clean))
        bar = float(schedule.alpha_bar[5])
        self.assertLess(abs(float(noisy.mean()) - np.sqrt(bar)*1.7), .01)
        self.assertLess(abs(float(noisy.var()) - (1-bar)), .01)
        op = MimoOperator(torch.eye(3, dtype=torch.complex128), 2)
        class NormalPrior(torch.nn.Module):
            def forward(self, image, time):
                scale = (1-schedule.alpha_bar[time-1]).sqrt()
                return image*scale[:, None, None, None]
        y = torch.randn(2, 2, 3, dtype=torch.complex128)
        a, trace = recover(NormalPrior(), schedule, op, y, .2, seed=51, refinement=2)
        b, _ = recover(NormalPrior(), schedule, op, y, .2, seed=51, refinement=2)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(trace[-1]["network_calls"], 12)
        self.assertTrue(torch.isfinite(a).all())

    def test_sure_scaled_risk_and_network_gradient(self):
        torch.manual_seed(5)
        h = .8*torch.randn(20000, 2, 2, 1, dtype=torch.float64)
        z = h+torch.randn_like(h)
        a, shrink = .5, .4
        estimator = lambda q: shrink*q/np.sqrt(a)
        corrected = risk(estimator, z, 1., probe=torch.ones_like(z))
        paper = risk(estimator, z, 1., variant="paper", probe=torch.ones_like(z))
        true_error = (shrink*z-h).square().flatten(1).mean(1)
        self.assertLess(abs(float(corrected.mean()-true_error.mean())), .02)
        self.assertAlmostEqual(float((paper-corrected).mean()),
                               2*shrink*(1/np.sqrt(a)-1), places=8)
        self.assertGreater(abs(float(paper.mean()-true_error.mean())), .25)
        schedule, network = VPSchedule(8), DenoisingCNN(width=8).double()
        index, target, delta = matched_time(schedule, .1)
        self.assertAlmostEqual(float(schedule.alpha_bar[index]), target+delta)
        batch = torch.randn(3, 2, 4, 4, dtype=torch.float64)
        loss = risk(lambda q: denoise(network, q, .1, index), batch, .1).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters()))
        self.assertAlmostEqual(signal_scale(50., 10, .5), 2.)
        with self.assertRaises(ValueError):
            signal_scale(1., 10, 1.)

    def test_empirical_lmmse_both_gram_sizes(self):
        torch.manual_seed(6)
        op = MimoOperator(torch.randn(3, 2, dtype=torch.complex128), 2)
        identity = torch.eye(12, dtype=torch.float64).reshape(12, 2, 2, 3)
        matrix = real_observation(op.forward(identity)).T
        y = torch.randn(2, 2, 2, dtype=torch.complex128)
        for count in (5, 19):
            images = torch.randn(count, 2, 2, 3, dtype=torch.float64)
            flat = images.flatten(1)/np.sqrt(count)
            basis = real_observation(op.forward(images))/np.sqrt(count)
            fitted = EmpiricalLMMSE(basis)
            actual = fitted.coefficients(y, .3)@flat
            covariance = flat.T@flat
            cross = covariance@matrix.T
            expected = (cross@torch.linalg.solve(matrix@cross+.15*torch.eye(8, dtype=matrix.dtype),
                                                 real_observation(y).T)).T
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    def test_new_training_resume_and_noise_prefix(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "validation", "test"):
                np.save(root/f"{split}.npy", np.ones((5, 2, 3), dtype=np.complex64))
            (root/"dataset.json").write_text("{}", encoding="utf-8")
            data = ChannelData(root, {"scenario": "mimo", "complex_scale": 1.,
                "counts": {s: 5 for s in ("train", "validation", "test")}, "channel_shape": [2, 3],
                "storage": "npy", "files": {s: f"{s}.npy" for s in ("train", "validation", "test")},
                "kind": "clean", "physics": {}, "known_noise_real_variance_phys": 0.})
            config = {"scenario": "mimo", "device": "cpu", "job": {
                "command": "train", "data_run_id": "fake", "sure_variant": "corrected"},
                "training": {"seed": 8, "validation_seed": 9, "batch_size": 2, "epochs": 2,
                "validation_every": 1, "validation_scenes": 3, "learning_rate": .001},
                "network": {"width": 8, "time_dim": 16}, "diffusion": {
                "steps": 8, "beta_start": .0001, "beta_end": .3}, "sure": {}}
            archives = [RunArchive(n, root/n) for n in ("full", "resumed")]
            for archive in archives:
                (archive.directory/"checkpoint").mkdir(parents=True)
            save = runner.atomic_torch
            def interrupt(path, payload):
                save(path, payload)
                raise KeyboardInterrupt()
            with patch.object(runner, "load_data", return_value=data):
                runner.train(archives[0], copy.deepcopy(config))
                with patch.object(runner, "atomic_torch", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.train(archives[1], copy.deepcopy(config))
                runner.train(archives[1], copy.deepcopy(config))
            full, resumed = [torch.load(a.directory/"checkpoint/training.pt", weights_only=True) for a in archives]
            for key in full["model"]:
                torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
            self.assertEqual(full["history"], resumed["history"])
            profiles = [json.loads((a.directory/"denoising_profile.json").read_text()) for a in archives]
            self.assertEqual(profiles[0], profiles[1])
            self.assertEqual([row["t"] for row in profiles[0]], [1, 2, 4, 6, 8])
            for row in profiles[0]:
                self.assertAlmostEqual(row["denoising_nmse_linear"],
                    2*(1-row["alpha_bar"])/row["alpha_bar"]*row["noise_prediction_mse"], places=5)
            data.close()
        settings = {"noise_seed": 91}
        small = runner.standard_noise(torch.zeros(2, 3, dtype=torch.complex64), settings, 1, "validation")
        large = runner.standard_noise(torch.zeros(5, 3, dtype=torch.complex64), settings, 1, "validation")
        torch.testing.assert_close(small, large[:2], rtol=0, atol=0)

    def test_independent_test_requires_matching_frozen_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"runs/reference"
            source.mkdir(parents=True)
            (source/"dataset.json").write_text("{}", encoding="utf-8")
            data = SimpleNamespace(directory=source, scenario="mimo", metadata={"kind":"clean", "physics":{}},
                                   counts={"train":8, "validation":4, "test":4})
            out = root/"check"
            out.mkdir()
            archive = RunArchive("check", out)
            settings = {"scene_limit":2, "validation_scenes":2}
            config = {"scenario":"mimo", "device":"cpu", "evaluation_run":settings,
                      "job":{"data_run_id":"reference", "train_run_id":[], "split":"test", "tuning_run_id":None}}
            with patch.object(runner, "PROJECT_ROOT", root), patch.object(runner, "load_data", return_value=data):
                with self.assertRaisesRegex(ValueError, "requires --tuning-run-id"):
                    runner.evaluate(archive, config)
                frozen = root/"runs/frozen"
                frozen.mkdir()
                for name, value in {
                    "status.json":{"status":"complete"},
                    "config_resolved.json":{"config":{"job":{"split":"validation"}, "evaluation_run":settings}},
                    "dependencies.json":runner.dependency_record(data, {}, {"reference":data}),
                    "tuning.json":{}}.items():
                    (frozen/name).write_text(json.dumps(value), encoding="utf-8")
                config["job"]["tuning_run_id"] = "frozen"
                with patch.object(runner, "build_operators", side_effect=RuntimeError("frozen accepted")):
                    with self.assertRaisesRegex(RuntimeError, "frozen accepted"):
                        runner.evaluate(archive, config)
                provenance = json.loads((out/"tuning_source.json").read_text())
                (frozen/"tuning.json").write_text('{"changed":true}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "tuning changed across test resume"):
                    runner.evaluate(archive, config)
                self.assertEqual(json.loads((out/"tuning_source.json").read_text()), provenance)
                invalid = copy.deepcopy(config)
                invalid["evaluation_run"]["scene_limit"] = 3
                with self.assertRaisesRegex(ValueError, "identical frozen"):
                    runner.evaluate(archive, invalid)


    def test_evaluation_resume_keeps_frozen_tuning(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"runs/reference"
            source.mkdir(parents=True)
            for split in ("train", "validation", "test"):
                np.save(source/f"{split}.npy", np.ones((5, 2, 3), dtype=np.complex64))
            (source/"dataset.json").write_text("{}", encoding="utf-8")
            data = ChannelData(source, {"scenario":"mimo", "complex_scale":1., "channel_shape":[2,3],
                "counts":{s:5 for s in ("train", "validation", "test")}, "kind":"clean", "physics":{},
                "split_seeds":{s:i for i,s in enumerate(("train", "validation", "test"))},
                "storage":"npy", "files":{s:f"{s}.npy" for s in ("train", "validation", "test")}})
            weight_path = root/"runs/weights/checkpoint"
            weight_path.mkdir(parents=True)
            (weight_path/"training.pt").write_bytes(b"frozen weight fingerprint")
            out = root/"check"
            (out/"checkpoint").mkdir(parents=True)
            archive = RunArchive("check", out)
            diffusion = {"steps":2, "beta_start":.001, "beta_end":.2}
            config = {"scenario":"mimo", "device":"cpu", "diffusion":diffusion,
                "data":{"batch_size":2}, "job":{"data_run_id":"reference", "train_run_id":["weights"],
                "split":"validation", "smoke":True}, "evaluation_run":{"scene_limit":2, "validation_scenes":2,
                "baseline_count":5, "guidance":[.5,1.], "lasso_factors":[.1], "snr_db":[0.],
                "refinement":1, "noise_seed":9, "posterior_seed":8, "bootstrap_seed":7, "lasso_steps":12}}
            loaded = (DenoisingCNN(width=8), VPSchedule(**diffusion), {"kind":"dm", "diffusion":diffusion}, data)
            operator = MimoOperator(torch.eye(3, dtype=torch.complex64), 2)
            write = runner.atomic_json
            def interrupt(path, payload):
                write(path, payload)
                if path.name == "eval_0_0_weights.json":
                    raise KeyboardInterrupt()
            with patch.object(runner, "PROJECT_ROOT", root), patch.object(runner, "load_data", return_value=data), \
                 patch.object(runner, "load_model", return_value=loaded), \
                 patch.object(runner, "build_operators", return_value=([("tiny", operator, None)], None)):
                with patch.object(runner, "atomic_json", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.evaluate(archive, config)
                frozen = json.loads((out/"tuning.json").read_text())["tiny:0:weights"]
                runner.evaluate(archive, config)
                validation = root/"runs/frozen"
                validation.mkdir()
                for name, value in {
                    "status.json":{"status":"complete"},
                    "config_resolved.json":{"config":config},
                    "dependencies.json":json.loads((out/"dependencies.json").read_text()),
                    "tuning.json":json.loads((out/"tuning.json").read_text())}.items():
                    (validation/name).write_text(json.dumps(value), encoding="utf-8")
                test_out = root/"independent"
                (test_out/"checkpoint").mkdir(parents=True)
                test_config = copy.deepcopy(config)
                test_config["job"].update(split="test", tuning_run_id="frozen")
                with patch.object(runner, "recover", wraps=runner.recover) as recovery, \
                     patch.object(runner, "lasso", wraps=runner.lasso) as sparse:
                    runner.evaluate(RunArchive("independent", test_out), test_config)
                self.assertEqual(recovery.call_count, 2)
                self.assertEqual(sparse.call_count, 2)
                self.assertEqual(json.loads((test_out/"tuning.json").read_text()),
                                 json.loads((validation/"tuning.json").read_text()))
                self.assertEqual(json.loads((test_out/"metrics_summary.json").read_text())["split"], "test")
            self.assertEqual(json.loads((out/"tuning.json").read_text())["tiny:0:weights"], frozen)
            self.assertEqual(json.loads((out/"metrics_summary.json").read_text())["rows"], 8)
            data.close()


if __name__ == "__main__":
    unittest.main()
