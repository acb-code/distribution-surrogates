import unittest
import numpy as np

from pyToolBox.rom import GenericROMBuilder


def convection_diffusion_1D(x, npts=100):

    _x = np.asarray(x).reshape(-1, 1)
    grid = np.linspace(-1, 1, npts)

    y = (1 - grid) * np.cos(3*np.pi*_x*(grid + 1))*np.exp(-_x*(grid + 1))

    return np.squeeze(y)


class TestGenericROM(unittest.TestCase):

    npts = 31
    ngrid = 100

    xall = np.linspace(1, np.pi, npts)

    xtrain = xall[::2]
    ytrain = convection_diffusion_1D(xtrain, ngrid)

    xtest = xall[1::2]
    ytest = convection_diffusion_1D(xtest, ngrid)

    def setUp(self):

        rom_builder = GenericROMBuilder(
            sm='rbf',
            sm_options={'regularize': False},
            dr='pca',
            dr_options={'ncomp': None}
        )

        rom = rom_builder.train(self.xtrain, self.ytrain)

        self.rom = rom

    def test_rmse(self):

        total_rmse, recon_rmse, regr_rmse  = self.rom.test_model(self.xtest, self.ytest)

        self.assertAlmostEqual(total_rmse, 0.017113050872054992)
        self.assertAlmostEqual(recon_rmse, 0.0022579639434187893)
        self.assertAlmostEqual(regr_rmse, 0.01696343446887342)


if __name__ == '__main__':
    unittest.main()
