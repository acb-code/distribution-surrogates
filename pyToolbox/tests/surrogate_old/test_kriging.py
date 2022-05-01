import unittest
import numpy as np

from old.surrogate.kriging import KrigingModel
from scipy.optimize import approx_fprime


def forrester1d(x):
    _x = x.ravel()
    y = (6 * _x - 2.) ** 2 * np.sin(12. * _x - 4.)
    return y


def rosenbrockNd(x):
    _x = np.atleast_2d(x)
    out = np.sum(100. * (_x[:, 1:] - _x[:, :-1]**2)**2 + (1 - _x[:, :-1])**2, axis=1)
    return out


class Test1D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.linspace(0., 1., 11)
        self.ytrain = forrester1d(self.xtrain)

        self.xtest = np.array([0.25, 0.5, 0.85])

        self.krg = KrigingModel(optimize=False)
        self.krg.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_eval(self):

        krg = self.krg

        ytest = np.array([-0.21136447, 0.90970964, -0.8093738])

        y = krg.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_std_error(self):

        krg = self.krg

        ytest = np.array([0.00335903, 0.00259317, 0.00607493])

        y = krg.std_error(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        krg = self.krg

        for xt in self.xtest:
            grad_fd1 = approx_fprime((xt,), krg.eval, self.dx)
            grad_fd2 = approx_fprime((xt,), krg.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = krg.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd, rtol=1e-4))

    def test_optim_mle(self):

        log_lkh = self.krg.log_likelihood

        krg_fast = KrigingModel(optimize=True, mode='fast')
        krg_fast.train(self.xtrain, self.ytrain)

        log_lkh_fast = krg_fast.log_likelihood

        krg_direct = KrigingModel(optimize=True, mode='direct')
        krg_direct.train(self.xtrain, self.ytrain)

        log_lkh_direct = krg_direct.log_likelihood

        self.assertGreater(log_lkh_fast, log_lkh)
        self.assertGreaterEqual(log_lkh_direct, log_lkh)

    def test_mle_grad(self):

        xcenter = self.krg.xcenter
        ycenter = self.krg.ycenter
        hparam = self.krg.hyper_param
        oparam = self.krg.Kernel.to_optim_param(hparam)
        Pmtx = np.hstack((np.ones((self.krg.npts, 1)), xcenter))

        lkh_ols, dlkh_ols = self.krg._obj_max_likelihood_ols(oparam, xcenter, grad=True)

        test_fun = lambda x: self.krg._obj_max_likelihood_ols(x, xcenter, grad=False)

        dlkh_ols_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_ols_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_ols_fd = 0.5 * (dlkh_ols_fd1 + dlkh_ols_fd2)

        self.assertTrue(np.allclose(dlkh_ols, dlkh_ols_fd, rtol=1e-3))

        lkh_gls, dlkh_gls = self.krg._obj_max_likelihood_gls(oparam, xcenter, ycenter, Pmtx,
                                                             grad=True, regularize=False)

        test_fun = lambda x: self.krg._obj_max_likelihood_gls(x, xcenter, ycenter, Pmtx,
                                                              grad=False, regularize=False)

        dlkh_gls_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_gls_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_gls_fd = 0.5 * (dlkh_gls_fd1 + dlkh_gls_fd2)

        self.assertTrue(np.allclose(dlkh_gls, dlkh_gls_fd, rtol=1e-3))


class Test2D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.mgrid[0:1:5j, 0:1:5j]
        self.xtrain = self.xtrain.reshape(2, -1).T

        self.xtest = np.array([[.1, .2],
                               [.4, .3],
                               [.5, .6],
                               [.8, .7]])

        self.ytrain = rosenbrockNd(self.xtrain)

        self.krg = KrigingModel(optimize=False)
        self.krg.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_eval(self):

        krg = self.krg

        ytest = np.array([2.42635665, 3.51227126, 12.04619213, 1.00423192])

        y = krg.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_interp(self):

        krg = self.krg
        y = krg.eval(self.xtrain)

        self.assertTrue(np.allclose(y, self.ytrain))

    def test_std_error(self):

        krg = self.krg

        ytest = np.array([1.38149053, 0.80690611, 0.66158412, 0.71512408])

        y = krg.std_error(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        krg = self.krg

        for xt in self.xtest:
            grad_fd1 = approx_fprime(xt, krg.eval, self.dx)
            grad_fd2 = approx_fprime(xt, krg.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = krg.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd))

    def test_optim_mle(self):

        log_lkh = self.krg.log_likelihood

        krg_fast = KrigingModel(optimize=True, mode='fast')
        krg_fast.train(self.xtrain, self.ytrain)

        log_lkh_fast = krg_fast.log_likelihood

        krg_direct = KrigingModel(optimize=True, mode='direct')
        krg_direct.train(self.xtrain, self.ytrain)

        log_lkh_direct = krg_direct.log_likelihood

        self.assertGreater(log_lkh_fast, log_lkh)
        self.assertGreaterEqual(log_lkh_direct, log_lkh)

    def test_mle_grad(self):
        xcenter = self.krg.xcenter
        ycenter = self.krg.ycenter
        hparam = self.krg.hyper_param
        oparam = self.krg.Kernel.to_optim_param(hparam)
        Pmtx = np.hstack((np.ones((self.krg.npts, 1)), xcenter))

        lkh_ols, dlkh_ols = self.krg._obj_max_likelihood_ols(oparam, xcenter, grad=True)

        test_fun = lambda x: self.krg._obj_max_likelihood_ols(x, xcenter, grad=False)

        dlkh_ols_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_ols_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_ols_fd = 0.5 * (dlkh_ols_fd1 + dlkh_ols_fd2)

        self.assertTrue(np.allclose(dlkh_ols, dlkh_ols_fd, rtol=1e-3))

        lkh_gls, dlkh_gls = self.krg._obj_max_likelihood_gls(oparam, xcenter, ycenter, Pmtx,
                                                             grad=True, regularize=False)

        test_fun = lambda x: self.krg._obj_max_likelihood_gls(x, xcenter, ycenter, Pmtx,
                                                              grad=False, regularize=False)

        dlkh_gls_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_gls_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_gls_fd = 0.5 * (dlkh_gls_fd1 + dlkh_gls_fd2)

        self.assertTrue(np.allclose(dlkh_gls, dlkh_gls_fd, rtol=1e-3))


class TestARD2D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.mgrid[0:1:5j, 0:1:5j]
        self.xtrain = self.xtrain.reshape(2, -1).T

        self.xtest = np.array([[.1, .2],
                               [.4, .3],
                               [.5, .6],
                               [.8, .7]])

        self.ytrain = rosenbrockNd(self.xtrain)

        self.krg = KrigingModel(optimize=False, kernel='ard_sqexp')
        self.krg.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_eval(self):

        krg = self.krg

        ytest = np.array([2.42635665, 3.51227126, 12.04619213, 1.00423192])

        y = krg.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_interp(self):

        krg = self.krg
        y = krg.eval(self.xtrain)

        self.assertTrue(np.allclose(y, self.ytrain))

    def test_std_error(self):

        krg = self.krg

        ytest = np.array([1.38149053, 0.80690611, 0.66158412, 0.71512408])

        y = krg.std_error(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        krg = self.krg

        for xt in self.xtest:
            grad_fd1 = approx_fprime(xt, krg.eval, self.dx)
            grad_fd2 = approx_fprime(xt, krg.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = krg.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd))

    def test_optim_mle(self):

        log_lkh = self.krg.log_likelihood

        krg_fast = KrigingModel(optimize=True, mode='fast')
        krg_fast.train(self.xtrain, self.ytrain)

        log_lkh_fast = krg_fast.log_likelihood

        krg_direct = KrigingModel(optimize=True, mode='direct')
        krg_direct.train(self.xtrain, self.ytrain)

        log_lkh_direct = krg_direct.log_likelihood

        self.assertGreater(log_lkh_fast, log_lkh)
        self.assertGreaterEqual(log_lkh_direct, log_lkh)

    def test_mle_grad(self):
        xcenter = self.krg.xcenter
        ycenter = self.krg.ycenter
        hparam = self.krg.hyper_param
        oparam = self.krg.Kernel.to_optim_param(hparam)
        Pmtx = np.hstack((np.ones((self.krg.npts, 1)), xcenter))

        lkh_ols, dlkh_ols = self.krg._obj_max_likelihood_ols(oparam, xcenter, grad=True)

        test_fun = lambda x: self.krg._obj_max_likelihood_ols(x, xcenter, grad=False)

        dlkh_ols_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_ols_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_ols_fd = 0.5 * (dlkh_ols_fd1 + dlkh_ols_fd2)

        self.assertTrue(np.allclose(dlkh_ols, dlkh_ols_fd, rtol=1e-3))

        lkh_gls, dlkh_gls = self.krg._obj_max_likelihood_gls(oparam, xcenter, ycenter, Pmtx,
                                                             grad=True, regularize=False)

        test_fun = lambda x: self.krg._obj_max_likelihood_gls(x, xcenter, ycenter, Pmtx,
                                                              grad=False, regularize=False)

        dlkh_gls_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_gls_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_gls_fd = 0.5 * (dlkh_gls_fd1 + dlkh_gls_fd2)

        self.assertTrue(np.allclose(dlkh_gls, dlkh_gls_fd, rtol=1e-3))


class TestRegularization(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)
        self.xtrain = np.linspace(0., 1., 30)
        self.ytrain = forrester1d(self.xtrain) + np.random.randn(30)

        self.krg = KrigingModel(optimize=True, regularize=False)
        self.krg.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_grad(self):

        xcenter = self.krg.xcenter
        ycenter = self.krg.ycenter
        hparam = self.krg.hyper_param
        oparam = self.krg.Kernel.to_optim_param(hparam)
        oparam = np.append(oparam, 0.)
        Pmtx = np.hstack((np.ones((self.krg.npts, 1)), xcenter))

        lkh_ols, dlkh_ols = self.krg._obj_max_likelihood_ols(oparam, xcenter, grad=True,
                                                             regularize=True)

        test_fun = lambda x: self.krg._obj_max_likelihood_ols(x, xcenter, grad=False,
                                                              regularize=True)

        dlkh_ols_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_ols_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_ols_fd = 0.5 * (dlkh_ols_fd1 + dlkh_ols_fd2)

        self.assertTrue(np.allclose(dlkh_ols, dlkh_ols_fd, rtol=1e-3))

        lkh_gls, dlkh_gls = self.krg._obj_max_likelihood_gls(oparam, xcenter, ycenter, Pmtx,
                                                             grad=True, regularize=True)

        test_fun = lambda x: self.krg._obj_max_likelihood_gls(x, xcenter, ycenter, Pmtx,
                                                              grad=False, regularize=True)

        dlkh_gls_fd1 = approx_fprime(oparam, test_fun, 1e-3)
        dlkh_gls_fd2 = approx_fprime(oparam, test_fun, -1e-3)
        dlkh_gls_fd = 0.5 * (dlkh_gls_fd1 + dlkh_gls_fd2)

        self.assertTrue(np.allclose(dlkh_gls, dlkh_gls_fd, rtol=1e-3))

    def test_optim(self):
        log_lkh = self.krg.log_likelihood

        krg_fast = KrigingModel(optimize=True,  regularize=True, mode='fast')
        krg_fast.train(self.xtrain, self.ytrain)

        log_lkh_fast = krg_fast.log_likelihood.item()

        krg_direct = KrigingModel(optimize=True, regularize=True, mode='direct')
        krg_direct.train(self.xtrain, self.ytrain)

        log_lkh_direct = krg_direct.log_likelihood.item()

        self.assertGreater(log_lkh_fast, log_lkh)
        self.assertGreaterEqual(log_lkh_direct, log_lkh)


if __name__ == '__main__':
    unittest.main()
