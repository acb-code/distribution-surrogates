import unittest
import numpy as np

from pyToolBox.surrogate import BridgeFunctionBuilder
from scipy.optimize import approx_fprime


def forrester1d_hi(x):
    return (6*x - 2)**2 * np.sin(12*x - 4)


def forrester1d_lo(x):
    return 0.5*forrester1d_hi(x) + 10*(x - 0.5) - 5.


class TestBridgeRBF(unittest.TestCase):

    trend_sm = 'rbf'
    trend_options = {
        'kernel': 'cubic'
    }
    bridge_sm = None
    bridge_options = None

    x_hi = np.array([0., 0.4, 0.6, 1.])
    y_hi = forrester1d_hi(x_hi)

    x_lo = np.linspace(0., 1., 11)
    y_lo = forrester1d_lo(x_lo)

    xtest = np.array([0.15, 0.25, 0.75, 0.85])
    ytest = np.array([0.4951688757204593,
                      0.505258833960033,
                      -1.1517742694346262,
                      3.8477735969633073])
    ytrue = forrester1d_hi(xtest)

    dx = 1e-4

    def setUp(self):

        mf_builder = BridgeFunctionBuilder(
            trend_sm=self.trend_sm,
            bridge_sm=self.bridge_sm,
            trend_options=self.trend_options,
            bridge_options=self.bridge_options,
        )
        hi_builder = mf_builder.trend_builder

        self.hi_builder = hi_builder
        self.mf_builder = mf_builder
        self.hi_model = hi_builder.train(self.x_hi, self.y_hi)
        self.mf_model = mf_builder.train(self.x_hi, self.y_hi, self.x_lo, self.y_lo)

    def test_eval(self):

        mf_model = self.mf_model
        ytest_mf = mf_model.eval(self.xtest)

        for y_p, y_t in zip(ytest_mf, self.ytest):
            self.assertTrue(np.allclose(y_p, y_t),
                            msg='Predicted value does not match test value.'
                                '\nPredicted output: {}'
                                '\nTrue output: {}'.format(y_p, y_t))

        hi_model = self.hi_model
        ytest_hi = hi_model.eval(self.xtest)

        for y_mf, y_hi, y_t in zip(ytest_mf, ytest_hi, self.ytrue):
            self.assertLessEqual(abs(y_mf - y_t), abs(y_hi - y_t),
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

        for xt, g in zip(self.xtest, grad):

            x_fd = np.atleast_1d(xt)

            grad_fd1 = approx_fprime(x_fd, mf_model.eval, self.dx)
            grad_fd2 = approx_fprime(x_fd, mf_model.eval, -self.dx)
            grad_fd = (grad_fd1 + grad_fd2) / 2

            self.assertTrue(np.allclose(g, grad_fd, rtol=1e-3),
                            msg='Model gradient does not match FD gradient.'
                                '\nAnalytical grad: {}'
                                '\nFinite Difference: {}'.format(g, grad_fd))

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


class TestBridgeKriging(TestBridgeRBF):

    trend_sm = 'kriging'
    trend_options = {
        'kernel': 'matern32',
        'gls_trend': False,
    }

    ytest = np.array([0.519871278730804,
                      0.5324989986751829,
                      -0.017364093507635125,
                      4.893469897184323])


if __name__ == '__main__':
    unittest.main()
