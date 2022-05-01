
import unittest
import numpy as np

from pyToolBox.surrogate import PolynomialBuilder
from scipy.optimize import approx_fprime


def planeNd(x):
    _x = np.atleast_2d(x)
    out = np.sum(_x, axis=1)
    return out


def sphereNd(x):
    _x = np.atleast_2d(x)
    out = np.sum(_x**2, axis=1)
    return out


# noinspection DuplicatedCode
class TestLinear1D(unittest.TestCase):

    def setUp(self):

        self.degree = 1
        np.random.seed(0)

        self.xtrain = np.random.rand(5, 1)
        self.xtest = np.random.rand(3, 1)

        self.ytrain = planeNd(self.xtrain)
        self.ytest = planeNd(self.xtest)

        self.builder = PolynomialBuilder(degree=self.degree, regularize=False)
        self.model = self.builder.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_eval(self):

        model = self.model

        for xi, yi in zip(self.xtest, self.ytest):
            yp = model.eval(xi)
            self.assertTrue(np.allclose(yp, yi),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(yp, yi))

    def test_grad(self):

        model = self.model

        grad = model.grad(self.xtest)
        _, grad_eval = model.eval(self.xtest, grad=True)

        self.assertTrue(np.allclose(grad, grad_eval),
                        msg='Eval and grad methods don\'t output identical gradients')

        for xt, g in zip(self.xtest, grad):
            x_fd = np.atleast_1d(xt)

            grad_fd1 = approx_fprime(x_fd, model.eval, self.dx)
            grad_fd2 = approx_fprime(x_fd, model.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            self.assertTrue(np.allclose(g, grad_fd, rtol=1e-3),
                            msg='Model gradient does not match FD gradient.'
                                '\nAnalytical grad: {}'
                                '\nFinite Difference: {}'.format(g, grad_fd))

    def test_loocv(self):

        ytrain = self.ytrain + np.random.randn(*self.ytrain.shape)
        model = self.builder.train(self.xtrain, ytrain)

        error_test = []

        for i in range(ytrain.size):

            _xi, _yi = self.xtrain[i], ytrain[i]
            _xtrain = np.delete(self.xtrain, i, axis=0)
            _ytrain = np.delete(ytrain, i, axis=0)

            _model = self.builder.train(_xtrain, _ytrain)
            _ytest = _model.eval(_xi)

            error_test.append(_ytest - _yi)

        loocv_test = np.sqrt(np.mean(np.array(error_test)**2))

        loocv = model.info['LOOCV']

        self.assertTrue(np.allclose(loocv, loocv_test),
                        msg='\nPredicted output: {}\nTrue output: {}'.format(loocv, loocv_test))

    def test_regularization(self):

        ytrain = self.ytrain + np.random.randn(*self.ytrain.shape)
        model = self.builder.train(self.xtrain, ytrain)

        builder_r = PolynomialBuilder(degree=self.degree, regularize=True)
        model_r = builder_r.train(self.xtrain, ytrain)

        self.assertGreater(model.info['LOOCV'], model_r.info['LOOCV'])
        self.assertLess(model.info['residual'], model_r.info['residual'])

    def test_zero_func(self):

        ytrain = np.zeros_like(self.ytrain)

        # noinspection PyBroadException
        try:
            with np.errstate(all='raise'):
                model = self.builder.train(self.xtrain, ytrain)
        except BaseException:
            self.fail('Training a null function raised an exception/warning.')

        ytest = model.eval(self.xtest)
        grad_ytest = model.grad(self.xtest)

        self.assertTrue(np.allclose(ytest, np.zeros_like(ytest)))
        self.assertTrue(np.allclose(grad_ytest, np.zeros_like(grad_ytest)))

    def test_under_sampled(self):

        nfactor = self.model.nfactor

        xtrain = self.xtrain[:nfactor-1]
        ytrain = self.ytrain[:nfactor-1]

        with self.assertWarns(UserWarning):
            self.builder.train(xtrain, ytrain)


# noinspection DuplicatedCode
class TestLinear5D(TestLinear1D):

    def setUp(self):

        self.degree = 1
        np.random.seed(0)

        self.xtrain = np.random.rand(10, 5)
        self.xtest = np.random.rand(5, 5)

        self.ytrain = planeNd(self.xtrain)
        self.ytest = planeNd(self.xtest)

        self.builder = PolynomialBuilder(degree=self.degree, regularize=False)
        self.model = self.builder.train(self.xtrain, self.ytrain)

        self.dx = 1e-6


# noinspection DuplicatedCode
class TestQuadratic1D(TestLinear1D):

    def setUp(self):

        self.degree = 2
        np.random.seed(0)

        self.xtrain = np.random.rand(15, 1)
        self.xtest = np.random.rand(4, 1)

        self.ytrain = sphereNd(self.xtrain)
        self.ytest = sphereNd(self.xtest)

        self.builder = PolynomialBuilder(degree=self.degree, regularize=False)
        self.model = self.builder.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_multi_output(self):

        for flag in (False, True):

            builder = PolynomialBuilder(degree=self.degree, regularize=flag)

            ytrain1 = self.ytrain[:, None]
            ytrain2 = planeNd(self.xtrain)[:, None]
            ytrain_mf = np.hstack((ytrain1, ytrain2))

            poly1 = builder.train(self.xtrain, ytrain1)
            ytest1 = poly1.eval(self.xtest)
            ygrad1 = poly1.grad(self.xtest)

            poly2 = builder.train(self.xtrain, ytrain2)
            ytest2 = poly2.eval(self.xtest)
            ygrad2 = poly2.grad(self.xtest)

            poly_mf = builder.train(self.xtrain, ytrain_mf)
            ytest_mf = poly_mf.eval(self.xtest)
            ygrad_mf = poly_mf.grad(self.xtest)

            flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
            self.assertTrue(flag_ytest, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))

            flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
            self.assertTrue(flag_ygrad, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))

            flag_loocv = np.allclose([poly1.info['LOOCV'],
                                      poly2.info['LOOCV']],
                                     poly_mf.info['LOOCV'])
            self.assertTrue(flag_loocv, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))


# noinspection DuplicatedCode
class TestQuadratic5D(TestQuadratic1D):

    def setUp(self):

        self.degree = 2
        np.random.seed(0)

        self.xtrain = np.random.rand(50, 5)
        self.xtest = np.random.rand(10, 5)

        self.ytrain = sphereNd(self.xtrain)
        self.ytest = sphereNd(self.xtest)

        self.builder = PolynomialBuilder(degree=self.degree, regularize=False)
        self.model = self.builder.train(self.xtrain, self.ytrain)

        self.dx = 1e-6


if __name__ == '__main__':
    unittest.main()
