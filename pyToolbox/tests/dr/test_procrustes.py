import unittest
import numpy as np
import scipy.linalg as la

from pyToolBox.dr import ProcrustesBuilder


def convection_diffusion_1D(x, npts=100):

    _x = np.asarray(x).reshape(-1, 1)
    grid = np.linspace(-1, 1, npts)

    y = (1 - grid) * np.cos(3*np.pi*_x*(grid + 1))*np.exp(-_x*(grid + 1))

    return np.squeeze(y)


npts_hi = 15
ngrid_hi = 500

npts_lo = 5 * npts_hi
ngrid_lo = 100

x_hi = np.linspace(1, np.pi, npts_hi)
y_hi = convection_diffusion_1D(x_hi, ngrid_hi)

x_lo = np.concatenate((x_hi, np.random.uniform(low=1, high=np.pi, size=npts_lo-npts_hi)))
y_lo = convection_diffusion_1D(x_lo, ngrid_lo)

test_error = 2.1586466142489066e-05


class TestProcrustes(unittest.TestCase):

    def setUp(self) -> None:

        builder = ProcrustesBuilder(ncomp=None)
        z_hi, z_lo, dr_hi, dr_lo = builder.train(y_hi, y_lo)

        self.z_hi = z_hi
        self.z_lo = z_lo
        self.dr_hi = dr_hi
        self.dr_lo = dr_lo

    def test_alignment(self):

        error = self.dr_hi.info['alignment_error']
        self.assertTrue(np.allclose(error, test_error))

    def test_reconstruction(self):

        z_test = self.dr_lo.compress(y_lo)

        self.assertTrue(np.allclose(z_test, self.z_lo),
                        msg='Latent variables from training and from model are different')


if __name__ == '__main__':
    unittest.main()
