import unittest
import numpy as np

from pyToolBox.surrogate import KrigingBuilder, MFKrigingBuilder


def forrester1d_hi(x):
    return (6*x - 2)**2 * np.sin(12*x - 4)


def forrester1d_lo1(x):
    return 0.5*forrester1d_hi(x) + 10*(x - 0.5) - 5.


def forrester1d_lo2(x):
    return 0.6*forrester1d_hi(x) + 9*(x - 0.5) - 4.


class TestMFKriging1D(unittest.TestCase):

    x_hi = np.array([0., 0.4, 0.6, 1.])
    y_hi = forrester1d_hi(x_hi)

    x_lo = np.linspace(0., 1., 11)
    y_lo = forrester1d_lo1(x_lo)

    xtest = np.array([0.15, 0.25, 0.75, 0.85])
    ytest = np.array([-1.00461351,
                      -0.18272577,
                      -6.03842019,
                      -0.66515109])
    ytrue = forrester1d_hi(xtest)

    kernel = 'sqexp'

    dx = 1e-4

    def setUp(self):

        hi_builder = KrigingBuilder(kernel=self.kernel)
        mf_builder = MFKrigingBuilder(kernel=self.kernel)

        self.hi_builder = hi_builder
        self.mf_builder = mf_builder
        self.hi_model = hi_builder.train(self.x_hi, self.y_hi)
        self.mf_model = mf_builder.train(self.x_hi, self.y_hi, self.x_lo, self.y_lo)

    def test_eval(self):

        mf_model = self.mf_model
        ytest_mf = mf_model.eval(self.xtest)

        # print(ytest_mf[:, None])

        for y_p, y_t in zip(ytest_mf, self.ytest):
            self.assertTrue(np.allclose(y_p, y_t),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_p, y_t))

        hi_model = self.hi_model
        ytest_hi = hi_model.eval(self.xtest)

        for y_mf, y_hi, y_t in zip(ytest_mf, ytest_hi, self.ytrue):
            self.assertTrue(np.all((y_mf - y_t)**2 <= (y_hi - y_t)**2),
                            msg='Multi-Fidelity prediction is worst than single-fidelity'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_mf, y_hi))

    def test_interp(self):

        mf_model = self.mf_model
        y = mf_model.eval(self.x_hi)

        for y_p, y_t in zip(y, self.y_hi):
            self.assertTrue(np.allclose(y_p, y_t),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_p, y_t))

    def test_grad(self):

        mf_model = self.mf_model

        grad = mf_model.grad(self.xtest)
        _, grad_eval = mf_model.eval(self.xtest, grad=True)

        self.assertTrue(np.allclose(grad, grad_eval),
                        msg='Eval and grad methods don\'t output identical gradients')

        for xt in self.xtest:
            x_fd = np.atleast_1d(xt)

            grad_fd1 = mf_model.eval(x_fd + self.dx)
            grad_fd2 = mf_model.eval(x_fd - self.dx)
            grad_fd = (grad_fd1 - grad_fd2) / (2 * self.dx)

            grad_ex = mf_model.grad(xt)

            self.assertTrue(np.allclose(grad_ex, grad_fd, rtol=1e-4),
                            msg='Model gradient does not match FD gradient.'
                                '\nAnalytical grad: {}'
                                '\nFinite Difference: {}'.format(grad_ex, grad_fd))

    def test_single_fidelity(self):

        x_hi = self.x_lo[::2]
        y_hi = self.y_lo[::2]

        mf_model = self.mf_builder.train(x_hi, y_hi, self.x_lo, self.y_lo)

        for xi, yi in zip(self.x_lo, self.y_lo):
            ytest = mf_model.eval(xi)
            self.assertTrue(np.allclose(ytest, yi),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(ytest, yi))


class TestMFKriging2DShared(TestMFKriging1D):

    x_hi = np.array([0., 0.4, 0.6, 1.])
    y_hi = forrester1d_hi(x_hi)

    x_lo = np.linspace(0., 1., 11)
    y_lo1 = forrester1d_lo1(x_lo)
    y_lo2 = forrester1d_lo2(x_lo)
    y_lo = np.stack((y_lo1, y_lo2), axis=-1)

    xtest = np.array([0.15, 0.25, 0.75, 0.85])
    ytest = np.array([-1.0106577 ,
                      -0.18576478,
                      -6.05289609,
                      -0.67691584])
    ytrue = forrester1d_hi(xtest)

    dx = 1e-4


class TestMFKriging2DNotShared(TestMFKriging2DShared):
    y_hi = np.stack((TestMFKriging2DShared.y_hi,
                     TestMFKriging2DShared.y_hi), axis=-1)


if __name__ == '__main__':
    unittest.main()
