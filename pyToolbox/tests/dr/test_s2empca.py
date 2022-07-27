import unittest
import numpy as np
import scipy.linalg as la

from pyToolBox.dr import PCABuilder, S2EMPCABuilder

np.random.seed(0)

NDIM_FULL_1 = 1000
NPTS_TRAIN_1 = 50
VAR_RATIO_1 = 0.8

NDIM_FULL_2 = 500
NPTS_TRAIN_2 = 200
VAR_RATIO_2 = 0.9

NPTS_TEST = 200

NCOMP = 4

latent_var_train = np.random.randn(NPTS_TRAIN_2, NDIM_FULL_1)
latent_var_test = np.random.randn(NPTS_TEST, NDIM_FULL_1)

var_vect_1 = VAR_RATIO_1 ** np.arange(NDIM_FULL_1)
var_vect_2 = VAR_RATIO_2 ** np.arange(NDIM_FULL_2)

_, BASIS_TRUE_1 = la.rq(np.random.randn(NDIM_FULL_1, NDIM_FULL_1), mode='economic')
_, BASIS_TRUE_2 = la.rq(np.random.randn(NDIM_FULL_2, NDIM_FULL_2), mode='economic')

DATA_TRAIN_1 = np.dot(latent_var_train[:NPTS_TRAIN_1, :] * np.sqrt(var_vect_1), BASIS_TRUE_1)
DATA_TRAIN_2 = np.dot(latent_var_train[:, :NDIM_FULL_2] * np.sqrt(var_vect_2), BASIS_TRUE_2)
DATA_TRAIN_1_FULL = np.dot(latent_var_train * np.sqrt(var_vect_1), BASIS_TRUE_1)

DATA_TEST = np.dot(latent_var_test * np.sqrt(var_vect_1), BASIS_TRUE_1)

RIC_TRAIN = 0.6694851108156097
NOISE_TRAIN = 0.0017762934406433779
ERROR_TRAIN = 0.6889080181106324
THETA_TRAIN = 0.3229631553259524


# noinspection DuplicatedCode
class TestS2EMPCA(unittest.TestCase):

    def setUp(self) -> None:

        pca_builder = PCABuilder(ncomp=NCOMP)
        _, pca = pca_builder.train(DATA_TRAIN_1, model=True)
        _, pca_full = pca_builder.train(DATA_TRAIN_1_FULL, model=True)

        s2empca_builder = S2EMPCABuilder(ncomp=NCOMP)
        ztrain_hi, ztrain_lo, s2empca_hi, s2empca_lo = \
            s2empca_builder.train(DATA_TRAIN_1, DATA_TRAIN_2, model=True)

        self.assertTrue(s2empca_hi.info['converged'],
                        msg='S2EMPCA builder did not converge')

        self.pca = pca
        self.pca_full = pca_full
        self.s2empca_hi = s2empca_hi
        self.s2empca_lo = s2empca_lo
        self.ztrain_hi = ztrain_hi
        self.ztrain_lo = ztrain_lo

    def test_basis(self):

        basis_pca = self.pca.basis
        basis_s2empca = self.s2empca_hi.basis
        basis_pca_full = self.pca_full.basis

        theta_pca = np.max(la.subspace_angles(basis_pca.T, basis_pca_full.T))
        theta_s2ppca = np.max(la.subspace_angles(basis_s2empca.T, basis_pca_full.T))

        self.assertAlmostEqual(theta_s2ppca, THETA_TRAIN,
                               places=6,
                               msg='\nMaximum principal angle: {}\nExpected value: {}'
                               .format(theta_s2ppca, THETA_TRAIN))

        self.assertLessEqual(theta_s2ppca, theta_pca,
                             msg='Semi-supervised results are worse than unsupervised results')

    def test_reconstruction(self):

        error_s2empca = self.s2empca_hi.test_model(DATA_TEST)
        error_pca = self.pca.test_model(DATA_TEST)

        self.assertAlmostEqual(error_s2empca, ERROR_TRAIN,
                               msg='\nReconstruction error: {}\nExpected value: {}'
                               .format(error_s2empca, ERROR_TRAIN))

        self.assertLessEqual(error_s2empca, error_pca,
                             msg='Semi-supervised results are worse than unsupervised results')

    # def test_noise(self):
    #
    #     self.assertIn('noise', self.s2empca_hi.info,
    #                   msg='\nModel noise information not found')
    #
    #     noise_s2empca = self.s2empca_hi.info['noise']
    #     noise_pca = self.pca.info['noise']
    #     noise_full = self.pca_full.info['noise']
    #
    #     self.assertAlmostEqual(noise_s2empca, NOISE_TRAIN,
    #                            msg='\nInfo Noise: {}\nExpected value: {}'
    #                            .format(noise_s2empca, NOISE_TRAIN))
    #
    #     self.assertLessEqual(abs(noise_full - noise_s2empca), abs(noise_full - noise_pca),
    #                          msg='Semi-supervised results are worse than unsupervised results')
    #
    #     comp_noise = la.norm(DATA_TRAIN_1 - self.s2empca_hi.expand(self.ztrain_hi)) ** 2
    #     comp_noise /= NPTS_TRAIN_1 * (NDIM_FULL_1 - NCOMP)
    #
    #     self.assertAlmostEqual(comp_noise, self.s2empca_hi.info['noise'],
    #                            msg='\nComputed Noise (High): {}\nExpected value: {}'
    #                            .format(comp_noise, NOISE_TRAIN))
    #
    #     comp_noise = la.norm(DATA_TRAIN_2 - self.s2empca_lo.expand(self.ztrain_lo)) ** 2
    #     comp_noise /= NPTS_TRAIN_2 * (NDIM_FULL_2 - NCOMP)
    #
    #     self.assertAlmostEqual(comp_noise, self.s2empca_lo.info['noise'],
    #                            msg='\nComputed Noise (Low): {}\nExpected value: {}'
    #                            .format(comp_noise, NOISE_TRAIN))

    # def test_latent(self):
    #
    #     self.assertTrue(np.allclose(self.ztrain_hi, self.ztrain_lo[:NPTS_TRAIN_1]),
    #                     msg='Inconsistent latent variable between high- and low-fidelity')

    def test_ric(self):

        self.assertIn('RIC', self.s2empca_hi.info,
                      msg='\nModel RIC information not found')

        ric_s2empca = self.s2empca_hi.info['RIC']
        ric_pca = self.pca.info['RIC']
        ric_full = self.pca_full.info['RIC']

        self.assertAlmostEqual(ric_s2empca, RIC_TRAIN,
                               msg='\nComputed RIC: {}\nExpected value: {}'
                               .format(ric_s2empca, RIC_TRAIN))

        self.assertLessEqual(abs(ric_full - ric_s2empca), abs(ric_full - ric_pca),
                             msg='Semi-supervised results are worse than unsupervised results')


if __name__ == "__main__":
    unittest.main()
