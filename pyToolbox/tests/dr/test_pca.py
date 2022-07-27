import unittest
import numpy as np
import scipy.linalg as la

from pyToolBox.dr import PCABuilder

np.random.seed(0)

NPTS_TRAIN = 100
NDIM_FULL = 100
VAR_RATIO = 0.75
NCOMP = 4

var_vect = VAR_RATIO ** np.arange(NDIM_FULL)

_, BASIS_TRUE = la.rq(np.random.randn(NDIM_FULL, NDIM_FULL), mode='economic')
DATA_TRAIN = np.dot(np.random.randn(NPTS_TRAIN, NDIM_FULL) * np.sqrt(var_vect), BASIS_TRUE)

RIC_TRAIN = 0.7014524843874832
TRAIN_ERROR = 0.10639731959103198
ERROR_TRAIN = 0.5463950179243187
THETA_TRAIN = 0.35298925043084645
VAR_TRAIN = [1.0761453665000564,
             0.735481499898079,
             0.4492876688436104,
             0.3988682481438132]


class TestSVDPCA(unittest.TestCase):

    def setUp(self):

        builder = PCABuilder(ncomp=NCOMP, method='svd')
        ztrain, model = builder.train(DATA_TRAIN, model=True)

        self.model = model
        self.ztrain = ztrain

    def test_basis(self):

        basis = self.model.basis
        theta_max = np.max(la.subspace_angles(basis.T, BASIS_TRUE[:NCOMP].T))

        self.assertAlmostEqual(theta_max, THETA_TRAIN,
                               msg='\nMaximum principal angle: {}\nExpected value: {}'
                               .format(theta_max, THETA_TRAIN))

    def test_compression(self):

        ztest = self.model.compress(DATA_TRAIN)

        self.assertTrue(np.allclose(ztest, self.ztrain),
                        msg='Latent variables from training and from model are different')

    def test_reconstruction(self):

        error = self.model.test_model(DATA_TRAIN)

        self.assertAlmostEqual(error, ERROR_TRAIN,
                               msg='\nReconstruction error: {}\nExpected value: {}'
                               .format(error, ERROR_TRAIN))

    def test_training_error(self):

        self.assertIn('error', self.model.info,
                      msg='\nModel noise information not found')

        info_error = self.model.info['error']

        comp_error = la.norm(DATA_TRAIN - self.model.expand(self.ztrain)) / DATA_TRAIN.size**0.5

        self.assertAlmostEqual(info_error, TRAIN_ERROR,
                               msg='\nInfo error: {}\nExpected value: {}'
                               .format(info_error, TRAIN_ERROR))

        self.assertAlmostEqual(comp_error, TRAIN_ERROR,
                               msg='\nComputed error: {}\nExpected value: {}'
                               .format(comp_error, TRAIN_ERROR))

    def test_ric(self):

        self.assertIn('RIC', self.model.info,
                      msg='\nModel RIC information not found')

        ric = self.model.info['RIC']

        self.assertAlmostEqual(ric, RIC_TRAIN,
                               msg='\nComputed RIC: {}\nExpected value: {}'
                               .format(ric, RIC_TRAIN))

    def test_variance(self):

        var = self.model.info['var']
        self.assertEqual(len(var), len(VAR_TRAIN))

        for i, (v1, v2) in enumerate(zip(var, VAR_TRAIN)):
            self.assertAlmostEqual(v1, v2,
                                   msg='\nComputed Variance[{}]: {}\nExpected value: {}'
                                   .format(i, v1, v2))


class TestLanczosPCA(TestSVDPCA):

    def setUp(self):

        builder = PCABuilder(ncomp=NCOMP, method='lanczos')
        ztrain, model = builder.train(DATA_TRAIN)

        self.model = model
        self.ztrain = ztrain


if __name__ == '__main__':
    unittest.main()
