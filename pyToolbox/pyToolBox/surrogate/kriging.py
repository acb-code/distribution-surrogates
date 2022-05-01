import numpy as np
import scipy.linalg as la
import warnings

from .common import BaseSurrogateOptim
from .kernel_sm import KernelSurrogateModel, KernelSurrogateBuilder
from .kernel_functions import PosDefKernel
from .polynomial import PolynomialModel, _polynomial_regressor_mtx


class KrigingBuilder(KernelSurrogateBuilder):

    _kernel_options = ['sqexp', 'matern32', 'matern52',
                       'ardsqexp', 'ardmatern32', 'ardmatern52']
    _trend_options = ['constant', 'linear', 'quadratic']

    def __init__(self,
                 kernel: str = 'matern32',
                 trend: str = 'linear',
                 optimize: bool = True,
                 regularize: bool = False,
                 gls_trend: bool = True,
                 maxiter: int = 500,
                 optim_restarts: int = 0,
                 normalize: bool = True,
                 warn: bool = True):

        super().__init__(kernel=kernel,
                         optimize=optimize,
                         regularize=regularize,
                         maxiter=maxiter,
                         optim_restarts=optim_restarts,
                         normalize=normalize)

        assert str(trend).lower() in self._trend_options

        self.trend = str(trend).lower()     # type: str
        self.gls_trend = bool(gls_trend)    # type: bool
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

        assert npts > 1

        kernel_func = self._get_kernel_func()   # type: PosDefKernel

        if hparam is not None:
            _hparam = np.atleast_1d(hparam)

            assert _hparam.ndim == 1
            kernel_func.check_hparam(_hparam, xdim)

        else:
            _hparam = kernel_func.get_default_hparam(xdim)

        info = {
            "model": "Kriging",
            "optimize": self.optimize,
            "regularize": self.regularize,
            "kernel": kernel_func.name,
        }

        if self.trend == 'constant':
            poly_degree = 0
        elif self.trend == 'linear':
            poly_degree = 1
        elif self.trend == 'quadratic':
            poly_degree = 2
        else:
            raise ValueError

        trend_mtx = _polynomial_regressor_mtx(_x, poly_degree)

        if trend_mtx.shape[0] <= trend_mtx.shape[1] and self.warn:
            warnings.warn("Training problem is under-determined.", UserWarning, 2)

        opt_prob = KrigingOptim(_x, _y, kernel_func, trend_mtx,
                                gls_trend=self.gls_trend,
                                regularize=self.regularize,
                                maxiter=self.maxiter,
                                optim_restarts=self.optim_restarts)

        if self.optimize:
            opt_results = opt_prob.optimize()
            info.update(
                maxiter=self.maxiter,
                optim_restarts=self.optim_restarts,
                optim_results=opt_results[0] if ydim == 1 else opt_results,
            )

        else:
            oparam = kernel_func.hyper_to_optim(_hparam)
            if self.regularize:
                oparam = np.append(oparam, -np.inf)
            opt_prob.eval(oparam)

        w_kernel = opt_prob.w_kernel
        w_trend = opt_prob.w_trend
        noise = opt_prob.noise
        _hparam = opt_prob.hparam
        log_likelyhood = opt_prob.log_likelyhood

        trend_model = PolynomialModel(xdim, ydim, poly_degree, w_trend)

        info.update(
            noise=noise.item() if ydim == 1 else noise,
            log_likelyhood=log_likelyhood.item() if ydim == 1 else log_likelyhood,
            trend_model=trend_model.name,
        )

        krg_model = KernelSurrogateModel(_x, w_kernel, kernel_func,
                                         trend=trend_model, info=info, hparam=_hparam,
                                         xoffset=_xoffset, xscale=_xscale,
                                         yoffset=_yoffset, yscale=_yscale)

        return krg_model


class KrigingOptim(BaseSurrogateOptim):

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 kernel_func: PosDefKernel,
                 trend_mtx: np.ndarray,
                 gls_trend: bool = True,
                 regularize: bool = False,
                 maxiter: int = 500,
                 optim_restarts: int = 0):

        super().__init__(maxiter=maxiter, optim_restarts=optim_restarts)

        self.x = x  # type: np.ndarray
        self.y = y  # type: np.ndarray

        self.npts = x.shape[0]  # type: int
        self.xdim = x.shape[1]  # type: int
        self.ydim = y.shape[1]  # type: int

        self.k_mtx = None       # type: [None, np.ndarray]
        self.k_mtx_u = None     # type: [None, np.ndarray]

        self.kernel_func = kernel_func  # type: [k_f.BaseKernel, k_f.PosDefKernel]
        self.regularize = regularize    # type: bool
        self.w_kernel = np.zeros((self.npts, self.ydim))  # type: np.ndarray

        self.gls_trend = gls_trend      # type: bool

        hbounds = self.kernel_func.get_optim_bounds(self.x)

        self.variance = np.zeros(self.ydim)                 # type: np.ndarray
        self.kernel_logdet = np.zeros(self.ydim)            # type: np.ndarray
        self.noise = np.zeros(self.ydim)                    # type: np.ndarray
        self.oparam = np.zeros((hbounds.shape[0], self.ydim))   # type: np.ndarray

        self._hbounds = hbounds                         # type: np.ndarray
        self._tiny = np.finfo(x.dtype).tiny             # type: float
        self._cond = self.npts * np.finfo(x.dtype).eps  # type: float

        self._noise = self.noise                # type: np.ndarray
        self._oparam = self.oparam              # type: np.ndarray
        self._k_logdet = self.kernel_logdet     # type: np.ndarray
        self._var = self.variance               # type: np.ndarray
        self._w_k = self.w_kernel               # type: np.ndarray
        self._y = self.y                        # type: np.ndarray

        self.w_trend = None     # type: [None, np.ndarray]
        self._w_t = None        # type: [None, np.ndarray]
        self.trend_mtx = None   # type: [None, np.ndarray]
        self._t_mtx = None      # type: [None, np.ndarray]
        self.tdim = 0           # type: int

        self._trend_init(trend_mtx)

    def _trend_init(self, t_mtx: np.ndarray) -> None:

        if self.gls_trend:
            self.tdim = t_mtx.shape[1]
            self.trend_mtx = t_mtx
            self.w_trend = np.zeros((self.tdim, self.ydim))

        else:
            self.w_trend = la.lstsq(t_mtx, self.y)[0]
            self.y -= np.dot(t_mtx, self.w_trend)
            self.trend_mtx = None
            self.tdim = 0

        self._w_t = self.w_trend
        self._t_mtx = self.trend_mtx

    @property
    def x0(self) -> np.ndarray:
        x0 = np.mean(self._hbounds, axis=1)
        if self.regularize:
            x0 = np.append(x0, np.log(0.1 * np.var(self.y[:, self.idx]) + self._tiny))
        return x0

    @property
    def bounds(self) -> np.ndarray:

        bounds = self._hbounds
        if self.regularize:
            log_noise_bounds = np.log(np.array([[0., self.y[:, self.idx].ptp() ** 2]]) + self._cond)
            bounds = np.append(bounds, log_noise_bounds, axis=0)

        return bounds

    @property
    def hparam(self) -> np.ndarray:
        return self.kernel_func.optim_to_hyper(self.oparam)

    @property
    def log_likelyhood(self) -> np.ndarray:
        return -0.5 * (self.npts * (np.log(self.variance + self._tiny)
                                    + np.log(2. * np.pi) + 1.) + self.kernel_logdet)

    def set_y_index(self, idx: [None, int]) -> None:
        super().set_y_index(idx)

        if self.idx is not Ellipsis:
            self._noise = self.noise[self.idx, None]
            self._oparam = self.oparam[:, self.idx]
            self._k_logdet = self.kernel_logdet[self.idx, None]
            self._var = self.variance[self.idx, None]
            self._w_t = self.w_trend[:, self.idx, None]
            self._w_k = self.w_kernel[:, self.idx, None]
            self._y = self.y[:, self.idx, None].copy()
        else:
            self._noise = self.noise
            self._oparam = self.oparam
            self._k_logdet = self.kernel_logdet
            self._var = self.variance
            self._w_t = self.w_trend
            self._w_k = self.w_kernel
            self._y = self.y

    def eval(self, dvar: np.ndarray) -> float:

        _dvar = np.atleast_1d(dvar)

        if self.dvar is None:
            self.dvar = _dvar.copy()
        else:
            self.dvar[:] = _dvar

        oparam = _dvar[:-1] if self.regularize else _dvar
        noise = np.exp(_dvar[-1]) if self.regularize else self._cond

        self.k_mtx = self.kernel_func.get_optim_matrix(self.x, oparam)
        self.k_mtx[np.diag_indices_from(self.k_mtx)] += noise

        self.k_mtx_u, cho_flag = la.cho_factor(self.k_mtx, lower=False)
        k_logdet = 2 * np.sum(np.log(self.k_mtx_u.diagonal()))

        # Generalized least-squares regression
        # Transform coordinate to get ordinary least-squares regression

        if self.gls_trend:

            y_k = la.solve_triangular(self.k_mtx_u, self._y, lower=cho_flag, trans=1, check_finite=False)

            t_mtx_k = la.solve_triangular(self.k_mtx_u, self._t_mtx, lower=cho_flag, trans=1, check_finite=False)
            t_mtx_k_q, t_mtx_k_r = la.qr(t_mtx_k, mode='economic')
            w_t = la.solve_triangular(t_mtx_k_r, np.dot(t_mtx_k_q.T, y_k), lower=False, check_finite=False)

            y_k -= t_mtx_k.dot(w_t)
            w_k = la.solve_triangular(self.k_mtx_u, y_k, lower=cho_flag, trans=0, check_finite=False)
            var = la.norm(y_k, axis=0)**2 / self.npts

            self._w_t[:] = w_t

        else:
            w_k = la.cho_solve((self.k_mtx_u, cho_flag), self._y, check_finite=False)
            var = np.sum(self._y * w_k, axis=0) / self.npts + self._tiny

        # condensed log likelihood
        obj = self.npts * np.log(var) + k_logdet

        self._noise[:] = noise
        self._k_logdet[:] = k_logdet
        self._oparam[:] = oparam
        self._w_k[:] = w_k
        self._var[:] = var

        return obj

    def grad(self, dvar: np.ndarray) -> np.ndarray:

        _dvar = np.atleast_1d(dvar)

        if np.any(_dvar != self.dvar):
            self.eval(_dvar)

        oparam = _dvar[:-1] if self.regularize else _dvar

        k_mtx_b = -la.lapack.dpotri(self.k_mtx_u, lower=0)[0]
        k_mtx_b = la.blas.dsyr(1. / self._var, self._w_k,
                               lower=0, a=k_mtx_b, overwrite_a=1)

        dobj = -self.kernel_func.get_optim_adjoint(self.x, oparam, self.k_mtx, k_mtx_b)

        if self.regularize:
            dnoise = -np.exp(_dvar[-1]) * np.sum(k_mtx_b.diagonal())
            dobj = np.append(dobj, dnoise)

        return dobj
