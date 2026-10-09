"""Physical complex-gradient, deterministic sampler and interrupted-training checks."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from thz_dma.estimators.diffusion import (ConditionalScoreNet, TorchDmaOperator,
    pack_channel, unpack_channel, posterior_samples, PreconditionedScoreNet,
    fourier_channel, denoised_likelihood_score, heun_flow, LowRankChannelPrior, SubspaceScoreModel,
    conditional_moment_score)
from thz_dma.estimators.planar_sparse import PlanarDmaTemplateOperator
from thz_dma.observations.multistrip import observe_multistrip_schedule
import run_direction1_diffusion as runner
from _run_support import RunArchive


class DiffusionTest(unittest.TestCase):
    def test_diagnostic_value_and_jacobian_are_separated(self):
        from diagnose_direction1_diffusion_components import MixedDenoiser
        class Prior:
            def project(self, x):
                return x
            def denoise(self, x, sigma):
                return .3*x
        class Core(torch.nn.Module):
            def forward(self, x, sigma, centers):
                return .2*x/sigma[:, None, None, None]
        for learned_value, value, derivative in [(True, .8, .3), (False, .3, .8)]:
            x = torch.randn(2, 2, 4, 1, dtype=torch.float64, requires_grad=True)
            sigma = torch.tensor([.5, 2.], dtype=x.dtype)
            model = MixedDenoiser(Core(), Prior(), learned_value=learned_value)
            clean = x-sigma[:, None, None, None]*model(x, sigma, torch.zeros(2, 9))
            torch.testing.assert_close(clean, value*x)
            gradient, = torch.autograd.grad(clean.sum(), x)
            torch.testing.assert_close(gradient, torch.full_like(x, derivative))

    def test_training_reference_observation_power(self):
        weights=np.array([[[[1.,2.,0.,0.]]]],dtype=complex)
        old=PlanarDmaTemplateOperator(np.ones(1),np.zeros((1,4,3)),weights,np.zeros(1),np.ones((1,1,1)))
        op=TorchDmaOperator(old,[0],tones=1)
        h=np.ones((1,1,4),dtype=np.complex64)
        data={'index':{'train':['a','b']},'real_component_scale':2.}
        with patch.object(runner,'read_example',side_effect=[(h,{}),(2*h,{})]):
            reference=runner.observation_signal_reference(Path('.'),data,{'one':op})
        self.assertAlmostEqual(reference['one'],(9+36)/2)

    def test_gaussian_guidance_and_fourier_preconditioning(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)
        n = 16
        weights = np.arange(1,9).reshape(1,1,1,8).astype(complex)/10
        old = PlanarDmaTemplateOperator(np.ones(1), np.zeros((1,8,3)),weights,np.zeros(1),np.ones((1,1,1)))
        op = TorchDmaOperator(old,[0],tones=1,dtype=torch.complex128)
        basis = torch.eye(n,dtype=torch.float64).reshape(n,2,8,1)
        observed=op.forward(basis).flatten(1)
        matrix=torch.cat((observed.real,observed.imag),dim=1).T
        y=torch.tensor([[[[.7+.4j]]]],dtype=torch.complex128)
        yreal=torch.tensor([.7,.4],dtype=torch.float64)
        covariance=torch.linalg.inv(torch.eye(n,dtype=torch.float64)+matrix.T@matrix/.15)
        mean=covariance@matrix.T@yreal/.15
        class NormalPrior(torch.nn.Module):
            def forward(self,x,sigma,centers):
                s=sigma[:,None,None,None]
                return s/(1+s*s)*x
        x=torch.randn(3,2,8,1,dtype=torch.float64)
        for sigma in (.01,.5,3.):
            score=denoised_likelihood_score(NormalPrior(),op,x,y,torch.zeros(3,9),.3,sigma)
            expected=-(x.flatten(1)-mean)@torch.linalg.inv(covariance+sigma*sigma*torch.eye(n))
            torch.testing.assert_close(score.flatten(1),expected)
        points=torch.cat((torch.eye(n),-torch.eye(n))).double()*np.sqrt(n)
        start=mean+points@torch.linalg.cholesky(covariance+9*torch.eye(n)).T
        samples,_=heun_flow(start,np.geomspace(3,.01,128),lambda x,s:
            -(x-mean)@torch.linalg.inv(covariance+s*s*torch.eye(n)))
        error=(samples-mean).T@(samples-mean)/len(samples)-covariance
        self.assertLess(float(torch.linalg.norm(error)/torch.linalg.norm(covariance)),.002)
        z=torch.randn(2,4,8,8)
        torch.testing.assert_close(fourier_channel(fourier_channel(z),inverse=True),z)
        self.assertAlmostEqual(float(z.square().sum()),float(fourier_channel(z).square().sum()),places=3)
        net=PreconditionedScoreNet(channels=4,width=4)
        sigma=torch.tensor([.1,3.])
        predicted=net(z,sigma,torch.zeros(2,9))
        s=sigma[:,None,None,None]
        torch.testing.assert_close(predicted,s/(1+s*s)*z)
        predicted.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_structured_conditional_gaussian(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(74)
        basis=(rng.normal(size=(3,1,1,8))+1j*rng.normal(size=(3,1,1,8))).astype(np.complex64)
        prior=LowRankChannelPrior(basis)
        weights=(rng.normal(size=(2,1,1,8))+1j*rng.normal(size=(2,1,1,8))).astype(complex)
        old=PlanarDmaTemplateOperator(np.ones(1),np.zeros((1,8,3)),weights,np.zeros(1),np.ones((1,2,1)))
        op=TorchDmaOperator(old,[0],tones=1)
        identity=torch.eye(16).reshape(16,2,8,1)
        mapped=op.forward(identity).flatten(1)
        a=torch.cat((mapped.real,mapped.imag),dim=1).T.double()
        r=(prior.u*prior.values)@prior.u.H
        covariance=.5*torch.cat((torch.cat((r.real,-r.imag),1),torch.cat((r.imag,r.real),1)),0).double()
        y=torch.tensor([[[[.5],[.3j]]]],dtype=torch.complex64)
        yr=torch.cat((y.real.flatten(),y.imag.flatten())).double()
        mean=covariance@a.T@torch.linalg.solve(a@covariance@a.T+.1*torch.eye(4),yr)
        post=covariance-covariance@a.T@torch.linalg.solve(a@covariance@a.T+.1*torch.eye(4),a@covariance)
        model=SubspaceScoreModel(None,prior)
        solve=prior.observation_solver(op,.2)
        x=torch.randn(2,2,8,1)
        for sigma in (.2,1.,8.):
            score=denoised_likelihood_score(model,op,x,y,torch.zeros(2,9),.2,sigma,solve)
            moment_score=conditional_moment_score(model,op,x,y,torch.zeros(2,9),.2,sigma,prior,solve)
            expected=-(x.flatten(1).double()-mean)@torch.linalg.inv(post+sigma*sigma*torch.eye(16))
            torch.testing.assert_close(score.flatten(1).double(),expected,rtol=3e-4,atol=2e-4)
            torch.testing.assert_close(moment_score.flatten(1).double(),expected,rtol=3e-4,atol=2e-4)
        class NonlinearCore(torch.nn.Module):
            def forward(self, x, sigma, centers):
                return .3*torch.tanh(x)
        nonlinear=SubspaceScoreModel(NonlinearCore(),prior)
        sigma=.7
        denoised=x-sigma*nonlinear(x,x.new_full((len(x),),sigma),torch.zeros(2,9))
        identity=torch.eye(16,dtype=covariance.dtype)
        local_cov=covariance@torch.linalg.solve(covariance+sigma*sigma*identity,sigma*sigma*identity)
        residual=yr[None]-denoised.flatten(1).double()@a.T
        conditional=denoised.flatten(1).double()+residual@torch.linalg.solve(
            a@local_cov@a.T+.1*torch.eye(4),a@local_cov)
        expected=(conditional-x.flatten(1).double())/sigma**2
        actual=conditional_moment_score(nonlinear,op,x,y,torch.zeros(2,9),.2,sigma,prior,solve)
        torch.testing.assert_close(actual.flatten(1).double(),expected,rtol=3e-4,atol=2e-4)
        torch.testing.assert_close(prior.project(prior.project(x)),prior.project(x),rtol=1e-5,atol=1e-5)
        latent_mean, latent_cov = prior.posterior_moments(op,y,.2)
        torch.testing.assert_close(prior.synthesize(latent_mean[None].cfloat()).flatten().double(),
                                   mean,rtol=2e-4,atol=1e-5)
        physical_cov = prior.u.to(torch.complex128)@latent_cov@prior.u.H.to(torch.complex128)
        real_cov = .5*torch.cat((torch.cat((physical_cov.real,-physical_cov.imag),1),
                                torch.cat((physical_cov.imag,physical_cov.real),1)),0)
        torch.testing.assert_close(real_cov,post,rtol=2e-4,atol=1e-5)
        settings=dict(sampler="guided_heun",sigma_min=.01,sigma_max=8.,levels=64,samples=4,
                      initialization="conditional_gaussian",antithetic=True)
        samples,_=posterior_samples(None,op,y,torch.zeros(1,9),.2,settings,seed=21,gaussian_prior=prior)
        torch.testing.assert_close(pack_channel(samples.mean(0)[None]).flatten().double(),mean,
                                   rtol=3e-4,atol=3e-5)
        settings['guidance']='conditional_moment_mean'
        moments,_=posterior_samples(None,op,y,torch.zeros(1,9),.2,settings,seed=21,gaussian_prior=prior)
        torch.testing.assert_close(moments,samples,rtol=3e-4,atol=3e-5)
        with self.assertRaises(ValueError):
            posterior_samples(None,op,y,torch.zeros(1,9),.2,settings,seed=21)

    def test_physical_operator_gradient_and_sampler(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(31)
        weights = rng.normal(size=(2, 3, 2, 8))+1j*rng.normal(size=(2, 3, 2, 8))
        relative = np.sum(abs(weights)**2, axis=-1).transpose(2, 0, 1)+.2
        old = PlanarDmaTemplateOperator(np.arange(3)+1., np.zeros((2, 8, 3)),
                                       weights, np.zeros(3), relative)
        op = TorchDmaOperator(old, [0, 2, 7], tones=8, dtype=torch.complex128)
        h = torch.tensor(rng.normal(size=(1, 8, 2, 8))+1j*rng.normal(size=(1, 8, 2, 8)))
        x = pack_channel(h)
        torch.testing.assert_close(unpack_channel(x), h)
        observation = observe_multistrip_schedule(weights, h.numpy()[0, [0, 2, 7]],
            pilot_energy_per_re=1, element_noise_variance=.1, rf_noise_variance=.02, rng=rng)
        expected = old.whiten(observation.clean).reshape(old.observation_shape)
        np.testing.assert_allclose(op.forward(x).numpy()[0], expected, atol=1e-12)
        y = torch.tensor(observation.observed)[None]
        self.assertAlmostEqual(float((op.forward(x).conj()*y).sum().real),
                               float((x*op.adjoint(y)).sum()), places=10)
        for sigma in (0, .4):
            z = x.clone().requires_grad_()
            factor = op.noise_factor(.3, sigma)
            residual = (y-op.forward(z)).permute(0, 3, 1, 2)[..., None]
            log_likelihood = -(residual.conj()*torch.cholesky_solve(residual, factor)).sum().real
            gradient, = torch.autograd.grad(log_likelihood, z)
            torch.testing.assert_close(gradient, op.likelihood_score(z, y, factor))
        with self.assertRaises(ValueError):
            op.noise_factor(.3, float("nan"))
        model = ConditionalScoreNet(channels=4, width=4)
        condition = torch.zeros(1, 9)
        out = model(x.float(), torch.ones(1), condition)
        out.square().sum().backward()
        self.assertEqual(out.shape, x.shape)
        op = TorchDmaOperator(old, [0, 2, 7], tones=8)
        settings = dict(sigma_min=.1, sigma_max=1., levels=2, steps_per_level=2,
                        samples=2, step_scale=.05)
        a, _ = posterior_samples(model, op, y.cfloat(), condition, .3, settings, seed=3)
        b, _ = posterior_samples(model, op, y.cfloat(), condition, .3, settings, seed=3)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(a).all())

    def test_training_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset.json").write_text("{}")
            dataset = {"index": {"train": ["a", "b"]}, "real_component_scale": 1.}
            config = {"job": {"data_run_id": "fake"}, "diffusion": {
                "device": "cpu", "network": {"width": 4},
                "posterior": {"sigma_min": .1, "sigma_max": 1.},
                "training": {"seed": 4, "learning_rate": .001, "steps": 3,
                    "batch_size": 1, "gradient_clip_norm": 1., "validation_every": 1}}}
            def batch(directory, names, scale, config, device):
                return torch.ones(len(names), 16, 8, 8), torch.zeros(len(names), 9)
            save = runner.atomic_torch
            def interrupt(path, payload):
                save(path, payload)
                raise KeyboardInterrupt()
            archives = [RunArchive(n, root/n) for n in ("full", "resumed")]
            for archive in archives:
                (archive.directory / "checkpoint").mkdir(parents=True)
            with patch.object(runner, "require_dataset", return_value=(root, dataset)), \
                    patch.object(runner, "load_batch", side_effect=batch), \
                    patch.object(runner, "validate_model", return_value=.5):
                runner.train(archives[0], copy.deepcopy(config))
                with patch.object(runner, "atomic_torch", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.train(archives[1], copy.deepcopy(config))
                runner.train(archives[1], copy.deepcopy(config))
            states = [torch.load(a.directory / "checkpoint/training.pt", weights_only=True) for a in archives]
            self.assertEqual(states[0]["history"], states[1]["history"])
            for key in states[0]["model"]:
                torch.testing.assert_close(states[0]["model"][key], states[1]["model"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
