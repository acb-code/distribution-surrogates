import unittest
import numpy as np
import pyToolBox.surrogate.kernel_functions as k_f

from pyToolBox.surrogate import KrigingBuilder
from pyToolBox.surrogate.kriging import KrigingOptim
from scipy.optimize import approx_fprime


def forrester1d(x):
    _x = x.ravel()
    y = (6 * _x - 2.) ** 2 * np.sin(12. * _x - 4.)
    return y


def rosenbrockNd(x):
    _x = np.atleast_2d(x)
    out = np.sum(100. * (_x[:, 1:] - _x[:, :-1]**2)**2 + (1 - _x[:, :-1])**2, axis=1)
    return out


def sphereNd(x):
    _x = np.atleast_2d(x)
    out = np.sum(_x**2, axis=1)
    return out


class TestSqExp1D(unittest.TestCase):

    kernel = 'sqexp'
    xtest = np.array([0.15, 0.25, 0.75, 0.85])
    ytest = np.array([-1.01557933,
                      -0.18355798,
                      -6.0520573 ,
                      -0.67911405])

    xtrain = np.linspace(0., 1., 11)
    ytrain = forrester1d(xtrain)

    kwargs = {'gls_trend': False}

    def setUp(self):
        self.builder = KrigingBuilder(kernel=self.kernel, **self.kwargs)
        self.krg = self.builder.train(self.xtrain, self.ytrain)
        self.dx = 1e-4

    def test_eval(self):

        rbf = self.krg
        y = rbf.eval(self.xtest)

        # print(y.reshape(-1, rbf.ydim))

        for y_p, y_t in zip(y, self.ytest):
            self.assertTrue(np.allclose(y_p, y_t, rtol=1e-4),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_p, y_t))

    def test_interp(self):

        rbf = self.krg
        y = rbf.eval(self.xtrain)

        for y_p, y_t in zip(y, self.ytrain):
            self.assertTrue(np.allclose(y_p, y_t, rtol=1e-4, atol=1e-6),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_p, y_t))

    def test_grad(self):

        krg = self.krg

        grad = krg.grad(self.xtest)
        _, grad_eval = krg.eval(self.xtest, grad=True)

        self.assertTrue(np.allclose(grad, grad_eval),
                        msg='Eval and grad methods don\'t output identical gradients')

        for xt, g in zip(self.xtest, grad):

            x_fd = np.atleast_1d(xt)

            grad_fd1 = approx_fprime(x_fd, krg.eval, self.dx)
            grad_fd2 = approx_fprime(x_fd, krg.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            self.assertTrue(np.allclose(g, grad_fd, rtol=1e-4),
                            msg='Model gradient does not match FD gradient.'
                                '\nAnalytical grad: {}'
                                '\nFinite Difference: {}'.format(g, grad_fd))

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

    def test_under_determined(self):

        xdim = self.krg.xdim
        npts = self.krg.npts

        np.random.seed(0)
        idx = np.random.choice(npts, size=xdim+1)

        xtrain = self.xtrain[idx]
        ytrain = self.ytrain[idx]

        with self.assertWarns(UserWarning):
            model = self.builder.train(xtrain, ytrain)

        for xi, yi in zip(xtrain, ytrain):
            ytest = model.eval(xi)
            self.assertTrue(np.allclose(ytest, yi),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(ytest, yi))


class TestMatern32_2D(TestSqExp1D):

    xtest = np.array([[.1, .2],
                      [.4, .3],
                      [.5, .6],
                      [.8, .7]])
    ytest = np.array([4.96477594,
                      1.6806012,
                      11.11846835,
                      3.20979784])

    xtrain = np.mgrid[0:1:4j, 0:1:4j]
    xtrain = xtrain.reshape(2, -1).T
    ytrain = rosenbrockNd(xtrain)

    kernel = 'matern32'


class TestARDMatern32_2D(TestMatern32_2D):

    ytest = np.array([ 4.55699264,
                       1.76847826,
                      11.55909581,
                       2.69580645])

    kernel = 'ardmatern32'


class TestMatern52_2D(TestMatern32_2D):

    ytest = np.array([ 5.78700789,
                       1.78942646,
                      11.45597751,
                       2.39712469])

    kernel = 'matern52'


class TestARDMatern52_2D(TestMatern32_2D):

    ytest = np.array([ 4.78564177,
                       1.94384004,
                      12.17553351,
                       1.48919607])

    kernel = 'ardmatern52'


class TestMatern52_3D(TestSqExp1D):

    xtest = np.array([[0.69759561, 0.72471007, 0.45077396],
                      [0.81773983, 0.01107362, 0.01705022],
                      [0.48495038, 0.2919652 , 0.17466634],
                      [0.59265214, 0.13648991, 0.2188263 ]])
    ytest = np.array([1.2150528205846514,
                      0.67000831,
                      0.35101721661992724,
                      0.4180335246912755])

    xtrain = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
    xtrain = xtrain.reshape(3, -1).T
    ytrain = sphereNd(xtrain)

    kernel = 'matern52'


class TestMultiKriging(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.mgrid[0:1:4j, 0:1:4j]
        self.xtrain = self.xtrain.reshape(2, -1).T

        self.xtest = np.array([[.1, .2],
                               [.4, .3],
                               [.5, .6],
                               [.8, .7]])

        self.ytrain1 = rosenbrockNd(self.xtrain)
        self.ytrain2 = sphereNd(self.xtrain)
        self.ytrain_mf = np.stack((self.ytrain1, self.ytrain2), axis=-1)

    def test_2D(self):

        for flag in (False, True):

            builder = KrigingBuilder(kernel='ardmatern32', gls_trend=flag)

            krg1 = builder.train(self.xtrain, self.ytrain1)
            ytest1 = krg1.eval(self.xtest)
            ygrad1 = krg1.grad(self.xtest)

            krg2 = builder.train(self.xtrain, self.ytrain2)
            ytest2 = krg2.eval(self.xtest)
            ygrad2 = krg2.grad(self.xtest)

            krg_mf = builder.train(self.xtrain, self.ytrain_mf)
            ytest_mf = krg_mf.eval(self.xtest)
            ygrad_mf = krg_mf.grad(self.xtest)

            flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
            self.assertTrue(flag_ytest, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))

            flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
            self.assertTrue(flag_ygrad, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))

            flag_loocv = np.allclose([krg1.info['log_likelyhood'],
                                      krg2.info['log_likelyhood']],
                                     krg_mf.info['log_likelyhood'])
            self.assertTrue(flag_loocv, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))

    def test_2D_no_optim(self):

        for flag in (False, True):

            builder = KrigingBuilder(optimize=False, gls_trend=flag)

            krg1 = builder.train(self.xtrain, self.ytrain1)
            ytest1 = krg1.eval(self.xtest)
            ygrad1 = krg1.grad(self.xtest)

            krg2 = builder.train(self.xtrain, self.ytrain2)
            ytest2 = krg2.eval(self.xtest)
            ygrad2 = krg2.grad(self.xtest)

            krg_mf = builder.train(self.xtrain, self.ytrain_mf)
            ytest_mf = krg_mf.eval(self.xtest)
            ygrad_mf = krg_mf.grad(self.xtest)

            flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
            self.assertTrue(flag_ytest, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))

            flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
            self.assertTrue(flag_ygrad, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))

            flag_loocv = np.allclose([krg1.info['log_likelyhood'],
                                      krg2.info['log_likelyhood']],
                                     krg_mf.info['log_likelyhood'])
            self.assertTrue(flag_loocv, msg='Test failed with {} flag'.format('GLS' if flag else 'OLS'))


class TestKernelOptim(unittest.TestCase):

    def setUp(self):

        xtrain = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
        xtrain = xtrain.reshape(3, -1).T
        ytrain = sphereNd(xtrain).reshape(-1, 1)

        self.x = (xtrain - xtrain.mean()) / xtrain.std()
        self.y = (ytrain - ytrain.mean()) / ytrain.std()
        self.dx = 1e-4

    def test_adjoint(self):

        kernel_list = KrigingBuilder._kernel_options
        trend_mtx = np.ones((self.x.shape[0], 1))

        for k in kernel_list:

            kernel_func = k_f.get_kernel_function(k)

            krg_obj = KrigingOptim(self.x, self.y, kernel_func, trend_mtx,
                                   gls_trend=False, regularize=True)
            krg_obj.set_y_index(0)

            bounds = krg_obj.bounds
            x0 = np.mean(bounds, axis=1)

            grad_fd1 = approx_fprime(x0, krg_obj.eval, self.dx)
            grad_fd2 = approx_fprime(x0, krg_obj.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = krg_obj.grad(x0)

            self.assertTrue(np.allclose(grad_ex, grad_fd, rtol=1e-4),
                            msg='\nAdjoint gradient for {} does not match FD gradient.'
                                '\nAnalytical grad: {}'
                                '\nFinite Difference: {}'
                            .format(kernel_func.name, grad_ex, grad_fd))


if __name__ == '__main__':
    unittest.main()
