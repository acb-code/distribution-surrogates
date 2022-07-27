import unittest
import numpy as np

from old.surrogate.rbf import RBFModel
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


def rmse(x1, x2):
    return np.sqrt(np.mean((x1 - x2)**2))


class TestCubic1D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.linspace(0., 1., 11)
        self.ytrain = forrester1d(self.xtrain)

        self.xtest = np.array([0.15, 0.25, 0.75, 0.85])

        self.rbf = RBFModel(kernel='cubic')
        self.rbf.train(self.xtrain, self.ytrain)

        self.dx = 1e-8

    def test_eval(self):

        rbf = self.rbf

        ytest = np.array([-0.9852891085305728,
                          -0.2443791556268593,
                          -6.042614791829305,
                          -0.3619555041757181])

        y = rbf.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        rbf = self.rbf

        for xt in self.xtest:
            grad_fd1 = approx_fprime((xt,), rbf.eval, self.dx)
            grad_fd2 = approx_fprime((xt,), rbf.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = rbf.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd))

    def test_loocv(self):

        error_test = []

        for i in range(self.ytrain.size):

            _xi, _yi = self.xtrain[i], self.ytrain[i]
            _xtrain = np.delete(self.xtrain, i)
            _ytrain = np.delete(self.ytrain, i)

            _rbf = RBFModel(kernel='cubic')
            _rbf.train(_xtrain, _ytrain)

            _ytest = _rbf.eval(_xi)

            error_test.append(_ytest - _yi)

        loocv_test = np.sqrt(np.mean(np.array(error_test)**2))

        loocv = self.rbf.loocv_error

        self.assertAlmostEqual(loocv, loocv_test)


class TestCubic2D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.mgrid[0:1:5j, 0:1:5j]
        self.xtrain = self.xtrain.reshape(2, -1).T

        self.xtest = np.array([[.1, .2],
                               [.4, .3],
                               [.5, .6],
                               [.8, .7]])

        self.ytrain = rosenbrockNd(self.xtrain)

        self.rbf = RBFModel(kernel='cubic')
        self.rbf.train(self.xtrain, self.ytrain)

        self.dx = 1e-8

    def test_eval(self):

        rbf = self.rbf

        ytest = np.array([3.9466245, 2.89306011, 12.17778049, 1.04322874])

        y = rbf.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_interp(self):

        rbf = self.rbf
        y = rbf.eval(self.xtrain)

        self.assertTrue(np.allclose(y, self.ytrain, atol=1e-7))

    def test_grad(self):

        rbf = self.rbf

        for xt in self.xtest:
            grad_fd1 = approx_fprime(xt, rbf.eval, self.dx)
            grad_fd2 = approx_fprime(xt, rbf.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = rbf.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd))


class TestCubic3D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
        self.xtrain = self.xtrain.reshape(3, -1).T

        self.xtest = np.array([[0.69759561, 0.72471007, 0.45077396],
                               [0.81773983, 0.01107362, 0.01705022],
                               [0.48495038, 0.2919652 , 0.17466634],
                               [0.59265214, 0.13648991, 0.2188263 ]])

        self.ytrain = sphereNd(self.xtrain)

        self.rbf = RBFModel(kernel='cubic')
        self.rbf.train(self.xtrain, self.ytrain)

        self.dx = 1e-8

    def test_eval(self):

        rbf = self.rbf

        ytest = np.array([1.21615291, 0.67926194, 0.35349066, 0.42338097])

        y = rbf.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        rbf = self.rbf

        for xt in self.xtest:
            grad_fd1 = approx_fprime(xt, rbf.eval, self.dx)
            grad_fd2 = approx_fprime(xt, rbf.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = rbf.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd))


class TestTPS1D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.linspace(0., 1., 11)
        self.ytrain = forrester1d(self.xtrain)

        self.xtest = np.array([0.15, 0.25, 0.75, 0.85])

        self.rbf = RBFModel(kernel='tps')
        self.rbf.train(self.xtrain, self.ytrain)

        self.dx = 1e-8

    def test_eval(self):

        rbf = self.rbf

        ytest = np.array([-0.9512228270905103,
                          -0.295914458670512,
                          -5.8000546461092615,
                          -0.29627227150375823])

        y = rbf.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        rbf = self.rbf

        for xt in self.xtest:
            grad_fd1 = approx_fprime((xt,), rbf.eval, self.dx)
            grad_fd2 = approx_fprime((xt,), rbf.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = rbf.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd))

    def test_loocv(self):

        error_test = []

        for i in range(self.ytrain.size):
            _xi, _yi = self.xtrain[i], self.ytrain[i]
            _xtrain = np.delete(self.xtrain, i)
            _ytrain = np.delete(self.ytrain, i)

            _rbf = RBFModel(kernel='tps')
            _rbf.train(_xtrain, _ytrain)

            _ytest = _rbf.eval(_xi)

            error_test.append(_ytest - _yi)

        loocv_test = np.sqrt(np.mean(np.array(error_test) ** 2))

        loocv = self.rbf.loocv_error

        self.assertAlmostEqual(loocv, loocv_test)


class TestTPS3D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
        self.xtrain = self.xtrain.reshape(3, -1).T

        self.xtest = np.array([[0.69759561, 0.72471007, 0.45077396],
                               [0.81773983, 0.01107362, 0.01705022],
                               [0.48495038, 0.2919652 , 0.17466634],
                               [0.59265214, 0.13648991, 0.2188263 ]])

        self.ytrain = sphereNd(self.xtrain)

        self.rbf = RBFModel(kernel='tps')
        self.rbf.train(self.xtrain, self.ytrain)

        self.dx = 1e-8

    def test_eval(self):

        rbf = self.rbf

        ytest = np.array([1.22724101, 0.69382895, 0.36734145, 0.44213486])

        y = rbf.eval(self.xtest)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        rbf = self.rbf

        for xt in self.xtest:
            grad_fd1 = approx_fprime(xt, rbf.eval, self.dx)
            grad_fd2 = approx_fprime(xt, rbf.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            grad_ex = rbf.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd))

    def test_loocv(self):

        error_test = []

        for i in range(self.ytrain.size):
            _xi, _yi = self.xtrain[i], self.ytrain[i]
            _xtrain = np.delete(self.xtrain, i, axis=0)
            _ytrain = np.delete(self.ytrain, i)

            _rbf = RBFModel(kernel='tps')
            _rbf.train(_xtrain, _ytrain)

            _ytest = _rbf.eval(_xi)

            error_test.append(_ytest - _yi)

        loocv_test = np.sqrt(np.mean(np.array(error_test) ** 2))

        loocv = self.rbf.loocv_error

        self.assertAlmostEqual(loocv, loocv_test, places=2)


class TestSqExp1D(unittest.TestCase):

    def setUp(self):

        self.xtrain = np.linspace(0., 1., 11)
        self.ytrain = forrester1d(self.xtrain)

        self.xtest = np.array([0.15, 0.25, 0.75, 0.85])

        self.rbf = RBFModel(kernel='sqexp', hyper_param=0.1)
        self.rbf.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_eval(self):

        rbf = self.rbf

        ytest = np.array([-0.97643008,
                          -0.21136447,
                          -5.98926062,
                          -0.8093738])

        y = rbf.eval(self.xtest)

        # print(y)

        self.assertTrue(np.allclose(y, ytest))

    def test_grad(self):

        rbf = self.rbf

        for xt in self.xtest:
            grad_fd1 = approx_fprime((xt,), rbf.eval, self.dx)
            grad_fd2 = approx_fprime((xt,), rbf.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2)/2

            grad_ex = rbf.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd, rtol=1e-3))

    def test_loocv(self):

        error_test = []

        for i in range(self.ytrain.size):
            _xi, _yi = self.xtrain[i], self.ytrain[i]
            _xtrain = np.delete(self.xtrain, i)
            _ytrain = np.delete(self.ytrain, i)

            _rbf = RBFModel(kernel='sqexp', hyper_param=0.1)
            _rbf.train(_xtrain, _ytrain)

            _ytest = _rbf.eval(_xi)

            error_test.append(_ytest - _yi)

        loocv_test = np.sqrt(np.mean(np.array(error_test) ** 2))

        loocv = self.rbf.loocv_error

        self.assertAlmostEqual(loocv, loocv_test)


class TestRegularization(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)
        self.xtrain = np.linspace(0., 1., 30)
        self.ytrain = forrester1d(self.xtrain) + np.random.randn(30)
        self.log_noise = -2.

        self.rbf = RBFModel(kernel='cubic', regularize=False, noise=np.exp(self.log_noise),
                            store_internal=True)
        self.rbf.train(self.xtrain, self.ytrain)

        self.dx = 1e-6

    def test_grad(self):

        K = self.rbf.internal['Kmtx']
        Q2 = self.rbf.internal['Pmtx_Q2']

        K_q = np.dot(Q2.T, np.dot(K, Q2))
        y_q = Q2.T.dot(self.ytrain)

        error, derror = self.rbf._obj_loocv_error(self.log_noise, K_q, y_q, Q2, True)

        test_fun = lambda x: self.rbf._obj_loocv_error(x, K_q, y_q, Q2, False)

        derror_fd1 = approx_fprime([self.log_noise], test_fun, self.dx)[0]
        derror_fd2 = approx_fprime([self.log_noise], test_fun, -self.dx)[0]
        derror_fd = 0.5*(derror_fd1 + derror_fd2)

        self.assertAlmostEqual(derror, derror_fd, places=6)

    def test_optim(self):

        loocv = self.rbf.loocv_error

        rbf2 = RBFModel(kernel='cubic', regularize=True, noise=np.exp(self.log_noise))
        rbf2.train(self.xtrain, self.ytrain)

        loocv_opt = rbf2.loocv_error

        self.assertLess(loocv_opt, loocv)


class TestMultiRBF(unittest.TestCase):

    def test_1D(self):

        xtrain = np.linspace(0., 1., 11)
        xtest = np.array([0.15, 0.25, 0.75, 0.85])

        ytrain1 = forrester1d(xtrain)
        ytrain2 = xtrain**2
        ytrain_mf = np.stack((ytrain1, ytrain2), axis=-1)

        rbf1 = RBFModel()
        rbf1.train(xtrain, ytrain1)
        ytest1 = rbf1.eval(xtest)
        ygrad1 = rbf1.grad(xtest)

        rbf2 = RBFModel()
        rbf2.train(xtrain, ytrain2)
        ytest2 = rbf2.eval(xtest)
        ygrad2 = rbf2.grad(xtest)

        rbf_mf = RBFModel()
        rbf_mf.train(xtrain, ytrain_mf)
        ytest_mf = rbf_mf.eval(xtest)
        ygrad_mf = rbf_mf.grad(xtest)

        flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
        self.assertTrue(flag_ytest)

        flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
        self.assertTrue(flag_ygrad)

        flag_loocv = np.allclose([rbf1.loocv_error, rbf2.loocv_error], rbf_mf.loocv_error)
        self.assertTrue(flag_loocv)

    def test_1D_noise(self):

        np.random.seed(6989)

        xtrain = np.linspace(0., 1., 30)
        xtest = np.array([0.15, 0.25, 0.75, 0.85])

        ytrain1 = forrester1d(xtrain) + np.random.randn(xtrain.size)
        ytrain2 = xtrain**2 + np.random.randn(xtrain.size)
        ytrain_mf = np.stack((ytrain1, ytrain2), axis=-1)

        rbf1 = RBFModel(regularize=True)
        rbf1.train(xtrain, ytrain1)
        ytest1 = rbf1.eval(xtest)
        ygrad1 = rbf1.grad(xtest)

        rbf2 = RBFModel(regularize=True)
        rbf2.train(xtrain, ytrain2)
        ytest2 = rbf2.eval(xtest)
        ygrad2 = rbf2.grad(xtest)

        rbf_mf = RBFModel(regularize=True)
        rbf_mf.train(xtrain, ytrain_mf)
        ytest_mf = rbf_mf.eval(xtest)
        ygrad_mf = rbf_mf.grad(xtest)

        flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
        self.assertTrue(flag_ytest)

        flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
        self.assertTrue(flag_ygrad)

        flag_loocv = np.allclose([rbf1.loocv_error, rbf2.loocv_error], rbf_mf.loocv_error)
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

        rbf1 = RBFModel(regularize=True)
        rbf1.train(xtrain, ytrain1)
        ytest1 = rbf1.eval(xtest)
        ygrad1 = rbf1.grad(xtest)

        rbf2 = RBFModel(regularize=True)
        rbf2.train(xtrain, ytrain2)
        ytest2 = rbf2.eval(xtest)
        ygrad2 = rbf2.grad(xtest)

        rbf_mf = RBFModel(regularize=True)
        rbf_mf.train(xtrain, ytrain_mf)
        ytest_mf = rbf_mf.eval(xtest)
        ygrad_mf = rbf_mf.grad(xtest)

        flag_ytest = np.allclose(np.stack((ytest1, ytest2), axis=-1), ytest_mf)
        self.assertTrue(flag_ytest)

        flag_ygrad = np.allclose(np.stack((ygrad1, ygrad2), axis=-1), ygrad_mf)
        self.assertTrue(flag_ygrad)

        flag_loocv = np.allclose([rbf1.loocv_error, rbf2.loocv_error], rbf_mf.loocv_error)
        self.assertTrue(flag_loocv)


if __name__ == '__main__':
    unittest.main()
