import numpy as np
import scipy.linalg as la

from .misc import _robust_solve, _svd_preproc
from .linear_model import LinearDRModel, LinearS2DRBuilder


class S2EMPCABuilder(LinearS2DRBuilder):

    def __init__(self,
                 ncomp: [None, int] = None,
                 atol: float = 1e-10,
                 rtol: float = 1e-8,
                 maxiter: int = 2000):

        super().__init__(ncomp)

        assert isinstance(self.ncomp, int)

        _atol = float(atol)
        assert _atol > 0.

        _rtol = float(rtol)
        assert 1. > _rtol > 0.

        assert maxiter > 0 and float(maxiter).is_integer()
        _maxiter = int(maxiter)

        self.atol = _atol           # type: float
        self.rtol = _rtol           # type: float
        self.maxiter = _maxiter     # type: int

    @property
    def is_shared_latent(self) -> bool:
        return True

    def train(self, data1: np.ndarray, data2: np.ndarray, model: bool = True) \
            -> [(np.ndarray, np.ndarray),
                (np.ndarray, np.ndarray, LinearDRModel, LinearDRModel)]:

        _data1 = self._check_data(data1)
        _data2 = self._check_data(data2)
        _model = bool(model)

        if self.ncomp is not None:
            assert self.ncomp < min(_data1.shape + _data2.shape)
            zdim = self.ncomp
        else:
            zdim = min(_data1.shape + _data2.shape)

        xmean = np.mean(_data1, axis=0)
        ymean = np.mean(_data2, axis=0)

        x, xsvals, xpca = _svd_preproc(_data1 - xmean)
        y, ysvals, ypca = _svd_preproc(_data2 - ymean)

        xsqnorm = np.sum(xsvals**2)
        ysqnorm = np.sum(ysvals**2)

        xnpts, xdim = x.shape
        ynpts, ydim = y.shape

        z = y[:, :zdim] * np.sqrt(ynpts)

        xscale = np.sqrt(xsqnorm / xdim / xnpts)
        yscale = np.sqrt(ysqnorm / ydim / ynpts)

        x *= xsvals / xscale
        y *= ysvals / yscale

        zs, zu = np.vsplit(z, [xnpts])
        ys, yu = np.vsplit(y, [xnpts])

        wx = np.zeros((zdim, xdim))
        wy = np.zeros((zdim, ydim))
        np.fill_diagonal(wx, xsvals[:zdim] / np.sqrt(xnpts) / xscale)
        np.fill_diagonal(wy, ysvals[:zdim] / np.sqrt(ynpts) / yscale)

        residuals = np.zeros(self.maxiter)
        z_old = z.copy()
        sqrt_z_size = np.sqrt(z.size)

        for i in range(self.maxiter):
            # E-Step
            tmp = np.dot(wy, wy.T)
            np.dot(yu, wy.T, out=zu)
            _robust_solve(tmp, zu.T, overwrite_b=True, assume_a='pos')

            tmp += np.dot(wx, wx.T)
            zs[:] = np.dot(x, wx.T) + np.dot(ys, wy.T)
            _robust_solve(tmp, zs.T, overwrite_a=True, overwrite_b=True, assume_a='pos')

            # M-Step
            tmp = np.dot(zs.T, zs, out=tmp)
            zs_pinv = _robust_solve(tmp, zs.T, assume_a='pos')
            np.dot(zs_pinv, x, out=wx)

            tmp += np.dot(zu.T, zu)
            z_pinv = _robust_solve(tmp, z.T, overwrite_a=True, assume_a='pos')
            np.dot(z_pinv, y, out=wy)

            # Check convergence
            residuals[i] = la.norm(z - z_old) / sqrt_z_size
            if residuals[i] < (self.atol + self.rtol * residuals[0]):
                residuals = residuals[:i + 1].copy()
                converged_flag = True
                break
            else:
                z_old[...] = z

        else:
            converged_flag = False

        tmp = np.dot(wx, wx.T)
        zx = np.dot(x, wx.T)
        _robust_solve(tmp, zx.T, overwrite_a=True, overwrite_b=True, assume_a='pos')

        tmp = np.dot(wy, wy.T)
        zy = np.dot(y, wy.T)
        _robust_solve(tmp, zy.T, overwrite_a=True, overwrite_b=True, assume_a='pos')

        if _model:

            xerror = la.norm(x - zx.dot(wx)) * xscale
            yerror = la.norm(y - zy.dot(wy)) * yscale

            xbasis = np.dot(wx * xscale, xpca)
            ybasis = np.dot(wy * yscale, ypca)

            xric = 1 - xerror**2 / xsqnorm
            yric = 1 - yerror**2 / ysqnorm

            xinfo = {
                'model': 'S2EMPCA',
                'RIC': xric,
                'error': xerror / _data1.size**0.5,
                'num_iter': residuals.size,
                'converged': converged_flag,
                'history': residuals.tolist(),
            }

            yinfo = xinfo.copy()
            yinfo.update({'RIC': yric, 'noise': yerror / _data2.size**0.5})

            model1 = LinearDRModel(xbasis, orthonorm=False, info=xinfo, xoffset=xmean)
            model2 = LinearDRModel(ybasis, orthonorm=False, info=yinfo, xoffset=ymean)

            return zx, zy, model1, model2

        else:
            return zx, zy,
