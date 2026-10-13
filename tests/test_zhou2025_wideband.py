"""Exact adjoint and real VP likelihood checks for full-digital wideband reception."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from zhou2025_repro.baselines import real_observation
from zhou2025_repro.data import decode, encode
from zhou2025_repro.wideband import DigitalArrayOperator


class DigitalArrayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_selected_full_elements_and_adjoint(self):
        torch.manual_seed(81)
        physical = torch.randn(2, 5, 2, 3, dtype=torch.complex128)
        operator = DigitalArrayOperator([4, 1], (5, 2, 3), repetitions=3)
        image = encode(physical, "dma")
        received = operator.forward(image)
        self.assertEqual(operator.image_shape, (4, 3, 5))
        self.assertEqual(received.shape, (2, 3, 2, 2, 3))
        self.assertEqual(received.dtype, torch.complex128)
        for repetition in range(3):
            torch.testing.assert_close(received[:, repetition], physical[:, [4, 1]])
        dual = torch.randn_like(received)
        adjoint = operator.adjoint(dual)
        torch.testing.assert_close((received.conj() * dual).sum().real, (image * adjoint).sum())
        state = image.detach().requires_grad_(True)
        gradient, = torch.autograd.grad((operator.forward(state).conj() * dual).sum().real, state)
        torch.testing.assert_close(adjoint, gradient, rtol=1e-12, atol=1e-12)
        recovered = decode(operator.adjoint(received), "dma")
        torch.testing.assert_close(recovered[:, [4, 1]], 3 * physical[:, [4, 1]])
        torch.testing.assert_close(recovered[:, [0, 2, 3]], torch.zeros_like(recovered[:, [0, 2, 3]]),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(operator.relative_variance, torch.ones(operator.observation_shape))

    def test_likelihood_matches_explicit_real_covariance(self):
        torch.manual_seed(82)
        for repetitions in (1, 3):
            operator = DigitalArrayOperator([2, 0], (3, 1, 2), repetitions=repetitions)
            size = 12
            basis = torch.eye(size, dtype=torch.float64).reshape(size, *operator.image_shape)
            matrix = real_observation(operator.forward(basis)).T
            image = torch.randn(2, *operator.image_shape, dtype=torch.float64)
            observation = torch.randn_like(operator.forward(image))
            for bar in (.07, .6, .99):
                state = image.clone().requires_grad_(True)
                residual = real_observation(observation) - state.flatten(1) @ matrix.T / bar**.5
                covariance = ((1 - bar) / bar * (matrix @ matrix.T)
                              + .15 * torch.eye(len(matrix), dtype=matrix.dtype))
                log_likelihood = -.5 * (residual * torch.linalg.solve(covariance, residual.T).T).sum()
                gradient, = torch.autograd.grad(log_likelihood, state)
                torch.testing.assert_close(operator.likelihood_score(image, observation, .3, bar),
                                           gradient, rtol=1e-10, atol=1e-10)

    def test_repetitions_use_independent_complex_noise_scale(self):
        torch.manual_seed(83)
        operator = DigitalArrayOperator([0], (2, 1, 1), repetitions=4)
        noise = torch.randn(12000, *operator.observation_shape, dtype=torch.complex128)
        self.assertLess(abs(float(noise.abs().square().mean()) - 1), .02)
        self.assertLess(abs(float(noise.real.var()) - .5), .02)
        averaged = noise.mean(dim=1)
        self.assertLess(abs(float(averaged.abs().square().mean()) - .25), .015)

    def test_invalid_acquisitions_and_noise(self):
        for indices in ([], [0, 0], [-1], [4], [0.5], [[0]], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                DigitalArrayOperator(indices, (4, 1, 2))
        for shape in ((4, 1), (4, 0, 2), (4., 1, 2), (4, True, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                DigitalArrayOperator([0], shape)
        for repetitions in (0, -1, 1.5, True):
            with self.subTest(repetitions=repetitions), self.assertRaises(ValueError):
                DigitalArrayOperator([0], (4, 1, 2), repetitions=repetitions)
        operator = DigitalArrayOperator([0], (4, 1, 2))
        image = torch.zeros(1, *operator.image_shape, dtype=torch.float64)
        observation = operator.forward(image)
        for variance, bar in ((0, .5), (-1, .5), (float("nan"), .5), (float("inf"), .5),
                              (.3, 0), (.3, 1), (.3, float("nan"))):
            with self.subTest(variance=variance, bar=bar), self.assertRaises(ValueError):
                operator.likelihood_score(image, observation, variance, bar)


if __name__ == "__main__":
    unittest.main()
