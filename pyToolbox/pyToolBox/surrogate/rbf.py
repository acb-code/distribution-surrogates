import numpy as np
import scipy.linalg as la
import warnings

from .common import BaseSurrogateOptim
from .kernel_sm import KernelSurrogateModel, KernelSurrogateBuilder
from .polynomial import PolynomialModel, _polynomial_regressor_mtx


class RBFBuilder(KernelSurrogateBuilder):

    _kernel_options = ['linear', 'cubic', 'tps', 'sqexp']
    _trend_options = ['constant', 'linear', 'quadratic']

    def __init__(self,
                 kernel: str = 'cubic',
                 trend: str = 'linear',
                 regularize: bool = False,
                 maxiter: int = 500,
                 optim_restarts: int = 0,
                 normalize: bool = True,
                 warn: bool = True):

        super().__init__(kernel=kernel,
                         optimize=regularize,
                         regularize=regularize,
                         maxiter=maxiter,
                         optim_restarts=optim_restarts,
                         normalize=normalize)

        assert str(trend).lower() in self._trend_options

        self.trend = str(trend).lower()     # type: str
        self.warn = bool(warn)              # type: bool

    def train(self, x: np.ndarray, y: np.ndarray,
              hparam: np.ndarray = None) -> KernelSurrogateModel:

        _x, _y = self._check_data(x, y, copy=True)

        if self.normalize:
            _x, _xoffset, _xscale = self._normalize_data(_x)
            _y, _yoffset, _yscale = self._normalize_data(_y)
        else:
            _xoffset = _xscale = None
            _yoffset = _yscale = None

        npts, xdim = _x.shape
        ydim = _y.shape[1]

        kernel_func = self._get_kernel_func()

        if kernel_func.has_hparam and hparam is not None:
            _hparam = np.atleast_1d(hparam)

            assert _hparam.ndim == 1
            kernel_func.check_hparam(_hparam, xdim)

        else:
            _hparam = kernel_func.get_default_hparam(xdim)

        info = {
            "model": "RBF",
            "regularize": self.regularize,
            "kernel": kernel_func.name,
        }

        if self.trend == 'constant':
            poly_degree = 0
            assert kernel_func.name.lower() in ('sqexp', 'linear')
        elif self.trend == 'linear':
            poly_degree = 1
        elif self.trend == 'quadratic':
            poly_degree = 2
        else:
            raise ValueError

        trend_mtx = _polynomial_regressor_mtx(_x, poly_degree)

        # noinspection PyTupleAssignmentBalance
        q_mtx, r_mtx = la.qr(trend_mtx, mode='full')
        r_mtx = r_mtx[:xdim + 1, :]
        q1_mtx, q2_mtx = np.hsplit(q_mtx, (xdim + 1,))

        if np.any(r_mtx.diagonal() == 0.) or q2_mtx.size < 1:

            if self.warn:
                warnings.warn("Training problem is under-determined.", UserWarning, 2)

            w_kernel = np.zeros_like(_y)

            # Solve polynomial regressor weights
            w_trend = np.dot(q1_mtx.T, _y)
            w_trend = la.lstsq(r_mtx, w_trend, overwrite_b=True)[0]

            trend_model = PolynomialModel(xdim, ydim, poly_degree, w_trend)
            info.update(trend_model=trend_model.name)

            rbf_model = KernelSurrogateModel(_x, w_kernel, kernel_func,
                                             trend=trend_model, info=info, hparam=_hparam,
                                             xoffset=_xoffset, xscale=_xscale,
                                             yoffset=_yoffset, yscale=_yscale)

            return rbf_model

        kernel_mtx = kernel_func.eval(_x, _x, hparam=_hparam)

        eps = np.finfo(kernel_mtx.dtype).eps
        max_eigval = np.max(np.sum(kernel_mtx, axis=1))
        kernel_mtx[np.diag_indices_from(kernel_mtx)] += eps * max_eigval

        kernel_q2_mtx = np.dot(q2_mtx.T, np.dot(kernel_mtx, q2_mtx))
        q2_y = np.dot(q2_mtx.T, _y)

        opt_prob = RBFOptim(kernel_q2_mtx, q2_y, q2_mtx,
                            maxiter=self.maxiter,
                            optim_restarts=self.optim_restarts)

        if self.regularize:
            opt_results = opt_prob.optimize()
            info.update(
                maxiter=self.maxiter,
                optim_restarts=self.optim_restarts,
                optim_results=opt_results[0] if ydim == 1 else opt_results,
            )

        else:
            opt_prob.eval(-np.inf)

        w_kernel = opt_prob.w_kernel
        noise = opt_prob.noise
        loocv = opt_prob.loocv

        if _yscale is not None:
            loocv *= _yscale

        # Solve polynomial regressor weights
        w_trend = q1_mtx.T.dot(_y - kernel_mtx.dot(w_kernel))
        w_trend = la.solve_triangular(r_mtx, w_trend, lower=False)

        trend_model = PolynomialModel(xdim, ydim, poly_degree, w_trend)

        info.update(
            noise=noise.item() if ydim == 1 else noise,
            LOOCV=loocv.item() if ydim == 1 else loocv,
            trend_model=trend_model.name,
        )

        rbf_model = KernelSurrogateModel(_x, w_kernel, kernel_func,
                                         trend=trend_model, info=info, hparam=_hparam,
                                         xoffset=_xoffset, xscale=_xscale,
                                         yoffset=_yoffset, yscale=_yscale)

        return rbf_model


class RBFOptim(BaseSurrogateOptim):

    def __init__(self,
                 k_q2_mtx: np.ndarray,
                 q2_y: np.ndarray,
                 q2_mtx: np.ndarray,
                 maxiter: int = 500,
                 optim_restarts: int = 0):

        super().__init__(maxiter=maxiter, optim_restarts=optim_restarts)

        self.k_q2_mtx = k_q2_mtx    # type: np.ndarray
        self.q2_mtx = q2_mtx        # type: np.ndarray
        self.q2_y = q2_y            # type: np.ndarray

        self.npts = q2_mtx.shape[0]     # type: int
        self.ydim = q2_y.shape[1]       # type: int

        self.w_kernel = np.zeros((self.npts, self.ydim))    # type: np.ndarray
        self.noise = np.zeros(self.ydim)                    # type: np.ndarray
        self.mse = np.zeros(self.ydim)                      # type: np.ndarray

        self._tiny = np.finfo(k_q2_mtx.dtype).tiny  # type: float

        self._noise = self.noise    # type: np.ndarray
        self._mse = self.mse        # type: np.ndarray
        self._w_k = self.w_kernel   # type: np.ndarray
        self._y = self.q2_y         # type: np.ndarray

        self.k_q2_cho = None        # type: [None, np.ndarray]
        self.w_kernel_q2 = None     # type: [None, np.ndarray]
        self.mse_mtx = None         # type: [None, np.ndarray]
        self.mse_vct = None         # type: [None, np.ndarray]

    @property
    def x0(self) -> np.ndarray:
        return np.atleast_1d(np.log(0.1 * self.q2_y[:, self.idx].var() + self._tiny))

    @property
    def bounds(self) -> np.ndarray:
        return np.log(np.array([[0., self.q2_y[:, self.idx].ptp() ** 2]]) + self._tiny)

    @property
    def loocv(self) -> np.ndarray:
        return np.sqrt(self.mse)

    def set_y_index(self, idx: [None, int]) -> None:
        super().set_y_index(idx)

        if self.idx is not Ellipsis:
            self._noise = self.noise[self.idx, None]
            self._mse = self.mse[self.idx, None]
            self._w_k = self.w_kernel[:, self.idx, None]
            self._y = self.q2_y[:, self.idx, None]
        else:
            self._noise = self.noise
            self._mse = self.mse
            self._w_k = self.w_kernel
            self._y = self.q2_y

    def eval(self, dvar: np.ndarray) -> float:

        _dvar = np.atleast_1d(dvar)

        if self.dvar is None:
            self.dvar = _dvar.copy()
        else:
            self.dvar[:] = _dvar

        noise = np.exp(_dvar)

        k_q2_mtx = self.k_q2_mtx.copy()
        k_q2_mtx[np.diag_indices_from(k_q2_mtx)] += noise

        k_q2_cho = la.cholesky(k_q2_mtx, overwrite_a=True, lower=False)
        w_k_q2 = la.cho_solve((k_q2_cho, False), self._y, overwrite_b=False)
        w_k = np.dot(self.q2_mtx, w_k_q2)

        mse_mtx = la.cho_solve((k_q2_cho, False), self.q2_mtx.T)
        mse_vct = np.einsum('ij, ji -> i', self.q2_mtx, mse_mtx)[:, None]

        self._mse[:] = np.mean((w_k / mse_vct) ** 2, axis=0)
        self._noise[:] = noise
        self._w_k[:] = w_k

        self.k_q2_cho = k_q2_cho
        self.w_kernel_q2 = w_k_q2
        self.mse_mtx = mse_mtx
        self.mse_vct = mse_vct

        return self._mse[:].mean()

    def grad(self, dvar: np.ndarray) -> np.ndarray:

        _dvar = np.atleast_1d(dvar)

        if np.any(_dvar != self.dvar):
            self.eval(_dvar)

        noise = np.exp(_dvar)

        w_kernel_fwd = la.cho_solve((self.k_q2_cho, False), -noise * self.w_kernel_q2)
        w_kernel_fwd = np.dot(self.q2_mtx, w_kernel_fwd)

        mse_vct_fwd = la.cho_solve((self.k_q2_cho, False), -noise * self.mse_mtx)
        mse_vct_fwd = np.einsum('ij, ji -> i', self.q2_mtx, mse_vct_fwd)[:, None]
        dobj = np.mean(2. * (self._w_k / self.mse_vct ** 3) *
                       (self.mse_vct * w_kernel_fwd - mse_vct_fwd * self._w_k), axis=0)

        return dobj
