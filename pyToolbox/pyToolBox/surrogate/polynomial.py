import numpy as np
import scipy.linalg as la
import warnings

from scipy.optimize import OptimizeResult, minimize_scalar
from .common import BaseSurrogateModel, BaseSurrogateBuilder, BaseSurrogateOptim


class PolynomialModel(BaseSurrogateModel):

    def __init__(self,
                 xdim: int,
                 ydim: int,
                 degree: int,
                 weights: np.ndarray,
                 info: dict = None,
                 xoffset: np.ndarray = None,
                 xscale: np.ndarray = None,
                 yoffset: np.ndarray = None,
                 yscale: np.ndarray = None):

        super().__init__()

        assert float(xdim).is_integer() and xdim > 0
        assert float(ydim).is_integer() and ydim > 0
        assert float(degree).is_integer() and 0 <= degree <= 2

        _xdim, _ydim, _degree = int(xdim), int(ydim), int(degree)

        if _degree == 0:
            nfactor = 1
        elif _degree == 1:
            nfactor = xdim + 1
        else:
            nfactor = (xdim + 1) * (xdim + 2) // 2

        _weights = np.asarray(weights)

        assert _weights.shape == (nfactor, ydim)

        _weights_0 = _weights[None, 0, :].copy()

        if _degree >= 1:
            _weights_1 = _weights[1:(xdim + 1), :].copy()
        else:
            _weights_1 = None

        if _degree == 2:
            idx = np.triu_indices(xdim)
            _weights_2 = np.zeros((_xdim, _xdim, _ydim))
            _weights_2[(idx[0], idx[1], ...)] = 0.5 * _weights[(xdim + 1):, :]
            _weights_2[(idx[0], idx[1], ...)] += 0.5 * _weights[(xdim + 1):, :]
        else:
            _weights_2 = None

        if info is not None:
            assert isinstance(info, dict)
            self.info.update(info)

        self.xdim = _xdim       # type: int
        self.ydim = _ydim       # type: int
        self.degree = _degree   # type: int
        self.nfactor = nfactor  # type: int

        self.weights_0 = _weights_0     # type: np.ndarray
        self.weights_1 = _weights_1     # type: [None, np.ndarray]
        self.weights_2 = _weights_2     # type: [None, np.ndarray]

        self._set_normalize(xoffset, xscale, yoffset, yscale)

    @property
    def name(self) -> str:
        if self.degree == 0:
            return 'ConstantPolynomial'
        elif self.degree == 1:
            return 'LinearPolynomial'
        else:
            return 'QuadraticPolynomial'

    def _eval(self, x: np.ndarray,
              grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:

        out = np.repeat(self.weights_0, x.shape[0], axis=0)

        if self.weights_1 is not None:
            out += np.dot(x, self.weights_1)

        if self.weights_2 is not None:
            out += np.einsum('ijk, ij -> ik', x.dot(self.weights_2), x, optimize=False)

        if grad:
            return out, self._grad(x)
        else:
            return out

    def _grad(self, x: np.ndarray) -> np.ndarray:

        if self.weights_1 is None and self.weights_2 is None:
            return np.zeros((x.shape + (self.ydim, )))

        gout = np.repeat(self.weights_1[None, ...], x.shape[0], axis=0)

        if self.weights_2 is not None:
            gout += 2 * np.dot(x, self.weights_2)

        return gout


class PolynomialBuilder(BaseSurrogateBuilder):

    def __init__(self,
                 degree: int = 1,
                 regularize: bool = False,
                 maxiter: int = 500,
                 optim_restarts: int = 0,
                 normalize: bool = True,
                 warn: bool = True):

        super().__init__(normalize=normalize)

        assert float(degree).is_integer() and 0 <= degree <= 2
        assert float(maxiter).is_integer()
        assert float(optim_restarts).is_integer()

        self.degree = int(degree)                   # type: int
        self.regularize = bool(regularize)          # type: bool
        self.maxiter = int(maxiter)                 # type: int
        self.optim_restarts = int(optim_restarts)   # type: int
        self.warn = bool(warn)                      # type: bool

    def train(self, x: np.ndarray, y: np.ndarray, hparam=None) -> PolynomialModel:

        _x, _y = self._check_data(x, y, copy=True)

        npts, xdim = _x.shape
        ydim = _y.shape[1]

        if self.normalize:
            _x, _xoffset, _xscale = self._normalize_data(_x)
            _y, _yoffset, _yscale = self._normalize_data(_y)
        else:
            _xoffset = _xscale = None
            _yoffset = _yscale = None

        info = {
            "model": "Polynomial",
            "regularize": self.regularize,
        }

        reg_mtx = _polynomial_regressor_mtx(_x, self.degree)

        if reg_mtx.shape[0] < reg_mtx.shape[1] and self.warn:
            warnings.warn("Training problem is under-determined.", UserWarning, 2)

        opt_prob = PolynomialObjective(reg_mtx, _y,
                                       maxiter=self.maxiter, optim_restarts=self.optim_restarts)

        if self.regularize:
            opt_results = opt_prob.optimize()
            info.update(
                maxiter=self.maxiter,
                optim_restarts=self.optim_restarts,
                optim_results=opt_results[0] if ydim == 1 else opt_results,
            )

        else:
            opt_prob.eval(-np.inf)

        noise = opt_prob.noise
        residual = opt_prob.residuals
        weights = opt_prob.weights
        loocv = opt_prob.loocv

        if _yscale is not None:
            loocv *= _yscale

        info.update(
            noise=noise.item() if ydim == 1 else noise,
            LOOCV=loocv.item() if ydim == 1 else loocv,
            residual=residual.item() if ydim == 1 else loocv,
        )

        return PolynomialModel(xdim, ydim, self.degree, weights,
                               info=info,
                               xoffset=_xoffset, xscale=_xscale,
                               yoffset=_yoffset, yscale=_yscale)


class PolynomialObjective(BaseSurrogateOptim):

    def __init__(self,
                 reg_mtx: np.ndarray,
                 y_vct: np.ndarray,
                 maxiter: int = 500,
                 optim_restarts: int = 0):

        super().__init__(maxiter=maxiter, optim_restarts=optim_restarts)

        # noinspection PyTupleAssignmentBalance
        u_mtx, d_vct, vh_mtx = la.svd(reg_mtx, full_matrices=False)

        self.npts = y_vct.shape[0]          # type: int
        self.nfeature = reg_mtx.shape[1]    # type: int
        self.ydim = y_vct.shape[1]          # type: int

        self.u_mtx = u_mtx      # type: np.ndarray
        self.vh_mtx = vh_mtx    # type: np.ndarray
        self.d_vct = d_vct      # type: np.ndarray
        self.y_vct = y_vct      # type: np.ndarray

        self.uy_mtx = np.dot(u_mtx.T, y_vct)    # type: np.ndarray

        self.noise = np.zeros(self.ydim)        # type: np.ndarray
        self.residuals = np.zeros(self.ydim)    # type: np.ndarray
        self.mse = np.zeros(self.ydim)          # type: np.ndarray

        self._tiny = np.finfo(y_vct.dtype).tiny     # type: float

        self._noise = self.noise    # type: np.ndarray
        self._res = self.residuals  # type: np.ndarray
        self._mse = self.mse        # type: np.ndarray
        self._uy = self.uy_mtx      # type: np.ndarray
        self._y = self.y_vct        # type: np.ndarray

    @property
    def bounds(self) -> list:
        d2max = np.max(self.d_vct) ** 2
        eps = np.finfo(self.y_vct.dtype).eps
        return [np.log(d2max * eps + self._tiny), np.log(1e3 * d2max + self._tiny)]

    @property
    def x0(self) -> float:
        return np.log(np.mean(self.d_vct) + self._tiny)

    @property
    def loocv(self) -> np.ndarray:
        return np.sqrt(self.mse)

    @property
    def weights(self) -> np.ndarray:

        weights = np.empty((self.nfeature, self.ydim))

        for i in range(self.ydim):
            d_noise = self.d_vct / (self.d_vct ** 2 + self.noise[i])
            weights[:, i] = np.dot(self.vh_mtx.T * d_noise[None, :], self.uy_mtx[:, i]).ravel()

        return weights

    def set_y_index(self, idx: [None, int]) -> None:
        super().set_y_index(idx)

        if self.idx is not Ellipsis:
            self._noise = self.noise[self.idx, None]
            self._res = self.residuals[self.idx, None]
            self._mse = self.mse[self.idx, None]
            self._uy = self.uy_mtx[:, self.idx, None]
            self._y = self.y_vct[:, self.idx, None]
        else:
            self._noise = self.noise
            self._res = self.residuals
            self._mse = self.mse
            self._uy = self.uy_mtx
            self._y = self.y_vct

    def eval(self, dvar: np.ndarray) -> float:

        _dvar = np.atleast_1d(dvar)

        if self.dvar is None:
            self.dvar = _dvar.copy()
        else:
            self.dvar[:] = _dvar

        noise = np.exp(dvar)
        self._noise[:] = noise

        d_noise = self.d_vct**2 / (self.d_vct**2 + noise)
        ud_mtx = self.u_mtx * d_noise[None, :]

        s_diag = np.einsum('ij, ij -> i', ud_mtx, self.u_mtx, optimize=False)[:, None]

        res = self._y - np.dot(ud_mtx, self._uy)
        error_vct = res / (1 - s_diag + self._tiny)
        self._res[:] = la.norm(res) / self.npts**0.5

        self._mse[:] = np.mean(error_vct ** 2, axis=0)

        return self._mse.sum()

    def optimize(self) -> list:

        opt_results = []

        for i in range(self.ydim):

            self.set_y_index(i)
            opt_bounds = self.bounds
            opt_yi = OptimizeResult(fun=np.inf, x=-np.inf)

            for ii in range(self.optim_restarts + 1):

                opt_tmp = minimize_scalar(self.eval,
                                          method='bounded',
                                          bounds=self.bounds)

                if opt_tmp.fun < opt_yi.fun or ii == 0:
                    opt_yi.update(opt_tmp)

            opt_results.append(opt_yi.copy())

            # re-run optimum
            self.eval(opt_yi.x)

        return opt_results


def _polynomial_regressor_mtx(x: np.ndarray, degree: int = 1):

    _x = np.atleast_2d(x)
    npts, ndim = _x.shape

    assert _x.ndim == 2
    assert float(degree).is_integer() and 0 <= degree <= 2

    if degree == 0:
        nfactor = 1
    elif degree == 1:
        nfactor = ndim + 1
    else:
        nfactor = (ndim + 1) * (ndim + 2) // 2

    mtx = np.ones((npts, nfactor))

    if degree >= 1:
        mtx[:, 1:(ndim + 1)] = _x

    if degree == 2:
        istart = ndim + 1

        for i in range(ndim):
            iend = istart + ndim - i
            mtx[:, istart:iend] = _x[:, i:] * _x[:, i, None]
            istart = iend

    return mtx
