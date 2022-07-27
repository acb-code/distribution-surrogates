import unittest
import numpy as np
import pyToolBox.surrogate.kernel_functions as k_f

from pyToolBox.surrogate import RBFBuilder
from pyToolBox.surrogate.rbf import RBFOptim
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


class TestCubic1D(unittest.TestCase):

    kernel = 'cubic'
    hparam = None
    xtest = np.array([0.15, 0.25, 0.75, 0.85])
    ytest = np.array([-0.9852891085305728,
                      -0.2443791556268593,
                      -6.042614791829305,
                      -0.3619555041757181])

    xtrain = np.linspace(0., 1., 11)
    ytrain = forrester1d(xtrain)

    kwargs = {
        'normalize': True
    }

    def setUp(self):
        self.builder = RBFBuilder(kernel=self.kernel, **self.kwargs)
        self.rbf = self.builder.train(self.xtrain, self.ytrain, hparam=self.hparam)
        self.dx = 1e-4

    def test_eval(self):

        rbf = self.rbf
        y = rbf.eval(self.xtest)

        # print(y.reshape(-1, 1))

        for y_p, y_t in zip(y, self.ytest):
            self.assertTrue(np.allclose(y_p, y_t),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_p, y_t))

    def test_interp(self):

        rbf = self.rbf
        y = rbf.eval(self.xtrain)

        for y_p, y_t in zip(y, self.ytrain):
            self.assertTrue(np.allclose(y_p, y_t),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_p, y_t))

    def test_grad(self):

        rbf = self.rbf

        grad = rbf.grad(self.xtest)
        _, grad_eval = rbf.eval(self.xtest, grad=True)

        self.assertTrue(np.allclose(grad, grad_eval),
                        msg='Eval and grad methods don\'t output identical gradients')

        for xt, g in zip(self.xtest, grad):
            x_fd = np.atleast_1d(xt)

            grad_fd1 = approx_fprime(x_fd, rbf.eval, self.dx)
            grad_fd2 = approx_fprime(x_fd, rbf.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            self.assertTrue(np.allclose(g, grad_fd, rtol=1e-3),
                            msg='Model gradient does not match FD gradient.'
                                '\nAnalytical grad: {}'
                                '\nFinite Difference: {}'.format(g, grad_fd))

    def test_loocv(self):

        error_test = []

        for i in range(self.ytrain.size):

            _xi, _yi = self.xtrain[i], self.ytrain[i]
            _xtrain = np.delete(self.xtrain, i, axis=0)
            _ytrain = np.delete(self.ytrain, i, axis=0)

            _rbf = self.builder.train(_xtrain, _ytrain, hparam=self.hparam)
            _ytest = _rbf.eval(_xi)

            error_test.append(_ytest - _yi)

        loocv_test = np.sqrt(np.mean(np.array(error_test)**2))

        loocv = self.rbf.info['LOOCV']

        self.assertTrue(np.allclose(loocv, loocv_test, rtol=5e-3, atol=5e-3),
                        msg='\nPredicted output: {}\nTrue output: {}'.format(loocv, loocv_test))

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

        xdim = self.rbf.xdim
        npts = self.rbf.npts

        np.random.seed(0)
        idx = np.random.choice(npts, size=xdim)

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


class TestLinear1D(TestCubic1D):

    kernel = 'linear'
    ytest = np.array([-0.6481519401260787,
                      -0.3276519198194525,
                      -4.777442239272156,
                      0.38140994912163695])


class TestTPS1D(TestCubic1D):

    kernel = 'tps'
    ytest = np.array([-0.9512228270905103,
                      -0.295914458670512,
                      -5.8000546461092615,
                      -0.29627227150375823])


class TestSqExp1D(TestCubic1D):

    kernel = 'sqexp'
    hparam = np.var(TestCubic1D.xtrain)
    ytest = np.array([-1.18722093,
                      -0.09462676,
                      -6.02424565,
                      -0.73483521])

    kwargs = {
        'normalize': False
    }


class TestCubic2D(TestCubic1D):

    xtest = np.array([[.1, .2],
                      [.4, .3],
                      [.5, .6],
                      [.8, .7]])
    ytest = np.array([3.9466245,
                      2.89306011,
                      12.17778049,
                      1.04322874])

    xtrain = np.mgrid[0:1:5j, 0:1:5j]
    xtrain = xtrain.reshape(2, -1).T
    ytrain = rosenbrockNd(xtrain)


class TestLinear2D(TestCubic2D):

    kernel = 'linear'
    ytest = np.array([3.8759873380369108,
                      2.375779211074903,
                      12.673726203608394,
                      2.6829701521620244])


class TestTPS2D(TestCubic2D):

    kernel = 'tps'
    ytest = np.array([3.72499568,
                      2.70361739,
                      12.13505309,
                      1.33504711])


class TestSqExp2D(TestCubic2D):

    kernel = 'sqexp'
    hparam = np.var(TestCubic2D.xtrain)
    ytest = np.array([ 3.72206842,
                       3.4478961 ,
                      12.9778357 ,
                       2.37007723])

    kwargs = {
        'normalize': False
    }


class TestCubic3D(TestCubic1D):
    xtest = np.array([[0.69759561, 0.72471007, 0.45077396],
                      [0.81773983, 0.01107362, 0.01705022],
                      [0.48495038, 0.2919652 , 0.17466634],
                      [0.59265214, 0.13648991, 0.2188263 ]])
    ytest = np.array([1.21615291, 0.67926194, 0.35349066, 0.42338097])

    xtrain = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
    xtrain = xtrain.reshape(3, -1).T
    ytrain = sphereNd(xtrain)


class TestLinear3D(TestCubic3D):

    kernel = 'linear'
    ytest = np.array([1.289190423237787,
                      0.6975877359884242,
                      0.4271272068636973,
                      0.49897209776923335])


class TestTPS3D(TestCubic3D):

    kernel = 'tps'
    ytest = np.array([1.22724101,
                      0.69382895,
                      0.36734145,
                      0.44213486])


class TestSqExp3D(TestCubic3D):

    kernel = 'sqexp'
    hparam = np.var(TestCubic3D.xtrain)
    ytest = np.array([1.48279885,
                      0.73175828,
                      0.59763047,
                      0.64903849])

    kwargs = {
        'normalize': False
    }


class TestRegularization(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)
        self.xtrain = np.linspace(0., 1., 30)
        self.ytrain = forrester1d(self.xtrain) + np.random.randn(30)

        builder = RBFBuilder(kernel='cubic', regularize=False)
        self.rbf_base = builder.train(self.xtrain, self.ytrain)

        builder.regularize = True
        self.rbf_reg = builder.train(self.xtrain, self.ytrain)

    def test_optim(self):

        loocv_base = self.rbf_base.info['LOOCV']
        loocv_opt = self.rbf_reg.info['LOOCV']

        self.assertLess(loocv_opt, loocv_base)

        self.assertTrue(self.rbf_reg.info['optim_results']['success'])


class TestMultiRBF(unittest.TestCase):

    def test_1D(self):

        xtrain = np.linspace(0., 1., 11)
        xtest = np.array([0.15, 0.25, 0.75, 0.85])

        ytrain1 = forrester1d(xtrain)
        ytrain2 = xtrain**2
        ytrain_mf = np.stack((ytrain1, ytrain2), axis=-1)

        builder = RBFBuilder()

        rbf1 = builder.train(xtrain, ytrain1)
        ytest1 = rbf1.eval(xtest)
        ygrad1 = rbf1.grad(xtest)

        rbf2 = builder.train(xtrain, ytrain2)
        ytest2 = rbf2.eval(xtest)
        ygrad2 = rbf2.grad(xtest)

        rbf_mf = builder.train(xtrain, ytrain_mf)
        ytest_mf = rbf_mf.eval(xtest)
        ygrad_mf = rbf_mf.grad(xtest)

        flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
        self.assertTrue(flag_ytest)

        flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
        self.assertTrue(flag_ygrad)

        flag_loocv = np.allclose([rbf1.info['LOOCV'], rbf2.info['LOOCV']], rbf_mf.info['LOOCV'])
        self.assertTrue(flag_loocv)

    def test_1D_noise(self):

        np.random.seed(6989)

        xtrain = np.linspace(0., 1., 30)
        xtest = np.array([0.15, 0.25, 0.75, 0.85])

        ytrain1 = forrester1d(xtrain) + np.random.randn(xtrain.size)
        ytrain2 = xtrain**2 + np.random.randn(xtrain.size)
        ytrain_mf = np.stack((ytrain1, ytrain2), axis=-1)

        builder = RBFBuilder(regularize=True)

        rbf1 = builder.train(xtrain, ytrain1)
        ytest1 = rbf1.eval(xtest)
        ygrad1 = rbf1.grad(xtest)

        rbf2 = builder.train(xtrain, ytrain2)
        ytest2 = rbf2.eval(xtest)
        ygrad2 = rbf2.grad(xtest)

        rbf_mf = builder.train(xtrain, ytrain_mf)
        ytest_mf = rbf_mf.eval(xtest)
        ygrad_mf = rbf_mf.grad(xtest)

        flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
        self.assertTrue(flag_ytest)

        flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
        self.assertTrue(flag_ygrad)

        flag_loocv = np.allclose([rbf1.info['LOOCV'], rbf2.info['LOOCV']], rbf_mf.info['LOOCV'])
        self.assertTrue(flag_loocv)

    def test_2D(self):

        xtrain = np.mgrid[0:1:5j, 0:1:5j]
        xtrain = xtrain.reshape(2, -1).T

        xtest = np.array([[.1, .2],
                          [.4, .3],
                          [.5, .6],
                          [.8, .7]])

        ytrain1 = rosenbrockNd(xtrain)
        ytrain2 = sphereNd(xtrain)
        ytrain_mf = np.stack((ytrain1, ytrain2), axis=-1)

        builder = RBFBuilder(regularize=True)

        rbf1 = builder.train(xtrain, ytrain1)
        ytest1 = rbf1.eval(xtest)
        ygrad1 = rbf1.grad(xtest)

        rbf2 = builder.train(xtrain, ytrain2)
        ytest2 = rbf2.eval(xtest)
        ygrad2 = rbf2.grad(xtest)

        rbf_mf = builder.train(xtrain, ytrain_mf)
        ytest_mf = rbf_mf.eval(xtest)
        ygrad_mf = rbf_mf.grad(xtest)

        flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
        self.assertTrue(flag_ytest)

        flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
        self.assertTrue(flag_ygrad)

        flag_loocv = np.allclose([rbf1.info['LOOCV'], rbf2.info['LOOCV']], rbf_mf.info['LOOCV'])
        self.assertTrue(flag_loocv)


class TestRbfOptim(unittest.TestCase):

    def setUp(self):

        xtrain = np.linspace(0., 1., 11).reshape(-1, 1)
        ytrain = forrester1d(xtrain).reshape(-1, 1)

        self.x = (xtrain - xtrain.mean()) / xtrain.std()
        self.y = (ytrain - ytrain.mean()) / ytrain.std()
        self.dx = 1e-4

    def test_adjoint(self):

        kernel_func = k_f.SqExpKernel()

        k_mtx = kernel_func.eval(self.x, self.x)
        q_mtx = np.identity(k_mtx.shape[0])

        rbf_obj = RBFOptim(k_mtx, self.y, q_mtx)
        rbf_obj.set_y_index(0)

        bounds = rbf_obj.bounds
        x0 = np.mean(bounds, axis=1)

        grad_fd1 = approx_fprime(x0, rbf_obj.eval, self.dx)
        grad_fd2 = approx_fprime(x0, rbf_obj.eval, -self.dx)
        grad_fd = (grad_fd1 + grad_fd2) / 2

        grad_ex = rbf_obj.grad(x0)

        self.assertTrue(np.allclose(grad_ex, grad_fd, rtol=1e-4),
                        msg='Adjoint gradient does not match FD gradient.'
                            '\nAnalytical grad: {}'
                            '\nFinite Difference: {}'.format(grad_ex, grad_fd))


if __name__ == '__main__':
    unittest.main()
