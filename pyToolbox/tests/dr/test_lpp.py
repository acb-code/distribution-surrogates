import unittest
import numpy as np
import scipy.linalg as la

from pyToolBox.dr import LPPBuilder

np.random.seed(0)

NPTS_TRAIN = 100
NDIM_FULL = 100
VAR_RATIO = 0.75
NCOMP = 4

var_vect = VAR_RATIO ** np.arange(NDIM_FULL)

_, BASIS_TRUE = la.rq(np.random.randn(NDIM_FULL, NDIM_FULL), mode='economic')
DATA_TRAIN = np.dot(np.random.randn(NPTS_TRAIN, NDIM_FULL) * np.sqrt(var_vect), BASIS_TRUE)


class TestLPPConstant(unittest.TestCase):

    RIC_TRAIN = 0.7014524843874834
    TRAIN_ERROR = 0.10639731959103202
    ERROR_TRAIN = 0.5463950179243188
    THETA_TRAIN = 0.35298925043084645

    def setUp(self):

        builder = LPPBuilder(ncomp=NCOMP, knn=10, weighted=False)
        ztrain, model = builder.train(DATA_TRAIN, model=True)

        self.model = model
        self.ztrain = ztrain

    def test_basis(self):

        basis = self.model.basis
        theta_max = np.max(la.subspace_angles(basis.T, BASIS_TRUE[:NCOMP].T))

        self.assertAlmostEqual(theta_max, self.THETA_TRAIN,
                               msg='\nMaximum principal angle: {}\nExpected value: {}'
                               .format(theta_max, self.THETA_TRAIN))

    def test_compression(self):

        ztest = self.model.compress(DATA_TRAIN)

        self.assertTrue(np.allclose(ztest, self.ztrain),
                        msg='Latent variables from training and from model are different')

    def test_reconstruction(self):

        error = self.model.test_model(DATA_TRAIN)

        self.assertAlmostEqual(error, self.ERROR_TRAIN,
                               msg='\nReconstruction error: {}\nExpected value: {}'
                               .format(error, self.ERROR_TRAIN))

    def test_training_error(self):

        self.assertIn('error', self.model.info,
                      msg='\nModel training error information not found')

        info_error = self.model.info['error']

        comp_error = la.norm(DATA_TRAIN - self.model.expand(self.ztrain)) / DATA_TRAIN.size**0.5

        self.assertAlmostEqual(info_error, self.TRAIN_ERROR,
                               msg='\nInfo error: {}\nExpected value: {}'
                               .format(info_error, self.TRAIN_ERROR))

        self.assertAlmostEqual(comp_error, self.TRAIN_ERROR,
                               msg='\nComputed error: {}\nExpected value: {}'
                               .format(comp_error, self.TRAIN_ERROR))


class TestLPPWeighted(TestLPPConstant):
    RIC_TRAIN = 0.7014524843874834
    TRAIN_ERROR = 0.10639731959103202
    ERROR_TRAIN = 0.5463950179243187
    THETA_TRAIN = 0.3529892504308464

    def setUp(self):
        builder = LPPBuilder(ncomp=NCOMP, knn=15, weighted=True, sig=2.)
        ztrain, model = builder.train(DATA_TRAIN, model=True)

        self.model = model
        self.ztrain = ztrain


if __name__ == '__main__':
    unittest.main()
