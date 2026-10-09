import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_zhou_reference_curves import angular,lasso,real_operator


class ReferenceTest(unittest.TestCase):
    def test_public_fixed_pilot_random_state(self):
        from diagnose_zhou_reference_pilots import author_pilots
        rng = np.random.RandomState(1)
        rng.binomial(1, .5, (3, 64, 64))
        rng.binomial(1, .5, (3, 64, 64))
        real = rng.binomial(1, .5, (2, 64, 38))
        imag = rng.binomial(1, .5, (2, 64, 38))
        expected = (2*real[0]-1+1j*(2*imag[0]-1))/np.sqrt(2)
        actual = author_pilots(1, 38, training_count=3, channel_count=2)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(abs(actual)**2, 1.)

    def test_complex_lasso_and_orthogonal_snr(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        h=torch.randn(200,2,8,dtype=torch.complex128)
        angle=angular(h)
        torch.testing.assert_close(angle.abs().square().sum(),h.abs().square().sum())
        b=torch.eye(8,dtype=torch.complex128)
        estimate,_,converged=lasso(angle,b,.2)
        expected=angle*(1-.2/angle.abs()).clamp_min(0)
        torch.testing.assert_close(estimate,expected)
        self.assertTrue(converged)
        # Orthogonal full pilots P P^H=Nt I give complex LS MSE=1/gamma.
        nt,gamma=8,100.
        p=torch.tensor(np.fft.fft(np.eye(nt),axis=0),dtype=torch.complex128)
        torch.testing.assert_close(p@p.H,nt*torch.eye(nt,dtype=p.dtype))
        noise=torch.randn(200,2,8,dtype=torch.complex128)*np.sqrt(nt/gamma)
        recovered=(h@p+noise)@p.H/nt
        self.assertLess(abs(float((recovered-h).abs().square().mean())-1/gamma),.001)
        a=p.T
        x=h[0,0]
        yr=a@x
        torch.testing.assert_close(real_operator(a)@torch.cat((x.real,x.imag)),torch.cat((yr.real,yr.imag)))
        real_a=real_operator(a)
        covariance=.5*torch.eye(2*nt,dtype=torch.float64)
        noise_variance=nt/gamma
        gain=covariance@real_a.T@torch.linalg.inv(real_a@covariance@real_a.T+noise_variance/2*torch.eye(2*nt))
        torch.testing.assert_close(gain@real_a,gamma/(1+gamma)*torch.eye(2*nt,dtype=torch.float64))


if __name__=='__main__':
    unittest.main()
