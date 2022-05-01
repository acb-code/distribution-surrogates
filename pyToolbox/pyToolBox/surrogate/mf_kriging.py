import numpy as np
import scipy.linalg as la
import warnings

from .common import BaseSurrogateModel
from .kernel_sm import KernelSurrogateModel, KernelMFSurrogateBuilder
from .kernel_functions import PosDefKernel
from .kriging import KrigingBuilder, KrigingOptim


class MFKrigingBuilder(KernelMFSurrogateBuilder):

    _kernel_options = ['sqexp', 'matern32', 'matern52',
                       'ardsqexp', 'ardmatern32', 'ardmatern52']

    def __init__(self,
                 kernel: str = 'matern32',
                 optimize: bool = True,
                 regularize: bool = False,
                 maxiter: int = 500,
                 optim_restarts: int = 0,
                 normalize: bool = True,
                 warn: bool = True,
                 shared_trend: bool = True,
                 trend_options: dict = None):

        super().__init__(kernel=kernel,
                         optimize=optimize,
                         regularize=regularize,
                         maxiter=maxiter,
                         optim_restarts=optim_restarts,
                         normalize=normalize)

        self.shared_trend = bool(shared_trend)  # type: bool
        self.warn = bool(warn)                  # type: bool

        if trend_options is None:
            self.trend_options = {}     # type: dict
        elif isinstance(trend_options, dict):
            _trend_options = trend_options.copy()
            self.trend_options = _trend_options
        else:
            raise ValueError

    def train(self,
              x_hi: np.ndarray, y_hi: np.ndarray,
              x_lo: np.ndarray, y_lo: np.ndarray,
              hparam_hi: np.ndarray = None,
              hparam_lo: np.ndarray = None) -> KernelSurrogateModel:

        _x_hi, _y_hi = self._check_data(x_hi, y_hi, copy=True)
        _x_lo, _y_lo = self._check_data(x_lo, y_lo, copy=True)

        npts_hi, xdim = _x_hi.shape
        npts_lo = _x_lo.shape[0]
        ydim_hi = _y_hi.shape[1]
        ydim_lo = _y_lo.shape[1]

        assert 1 <= npts_hi <= npts_lo
        assert xdim == _x_lo.shape[1]

        if not self.shared_trend:
            assert ydim_hi == ydim_lo

        if self.normalize:
            _x_hi, _xoffset, _xscale = self._normalize_data(_x_hi)
            _x_lo -= _xoffset
            _x_lo /= _xscale

            _y_hi, _yoffset, _yscale = self._normalize_data(_y_hi)
            _y_lo, _, _ = self._normalize_data(_y_lo)

        else:
            _xoffset = _xscale = None
            _yoffset = _yscale = None

        assert 1 < npts_hi < npts_lo

        kernel_func = self._get_kernel_func()  # type: PosDefKernel

        if hparam_hi is not None:
            _hparam_hi = np.atleast_1d(hparam_hi)

            assert _hparam_hi.ndim == 1
            kernel_func.check_hparam(_hparam_hi, xdim)

        else:
            _hparam_hi = kernel_func.get_default_hparam(xdim)

        if hparam_lo is not None:
            _hparam_lo = np.atleast_1d(hparam_lo)

            assert _hparam_lo.ndim == 1
            kernel_func.check_hparam(_hparam_lo, xdim)

        else:
            _hparam_lo = kernel_func.get_default_hparam(xdim)

        info = {
            "model": "MFKriging",
            "optimize": self.optimize,
            "regularize": self.regularize,
            "kernel": kernel_func.name,
        }

        trend_options = {
            'kernel': self.kernel,
            'optimize': self.optimize,
            'regularize': self.regularize,
            'maxiter': self.maxiter,
            'optim_restarts': self.optim_restarts,
        }
        trend_options.update(self.trend_options)
        trend_options.update(normalize=False)

        krg_builder = KrigingBuilder(**trend_options)

        krg_lo = krg_builder.train(_x_lo, _y_lo, hparam=hparam_lo)
        trend_mtx = krg_lo.eval(_x_hi).reshape(npts_hi, ydim_lo)

        if trend_mtx.shape[0] <= trend_mtx.shape[1] and self.warn and self.shared_trend:
            warnings.warn("Training problem is under-determined.", UserWarning, 2)

        opt_prob = MFKrigingOptim(_x_hi, _y_hi, kernel_func, trend_mtx,
                                  shared_trend=self.shared_trend,
                                  regularize=self.regularize,
                                  maxiter=self.maxiter,
                                  optim_restarts=self.optim_restarts)

        if self.optimize:
            opt_results = opt_prob.optimize()
            info.update(
                maxiter=self.maxiter,
                optim_restarts=self.optim_restarts,
                optim_results=opt_results[0] if ydim_hi == 1 else opt_results,
            )

        else:
            oparam = kernel_func.hyper_to_optim(_hparam_hi)
            if self.regularize:
                oparam = np.append(oparam, -np.inf)
            opt_prob.eval(oparam)

        w_kernel = opt_prob.w_kernel
        w_trend = opt_prob.w_trend
        noise = opt_prob.noise
        _hparam = opt_prob.hparam
        log_likelyhood = opt_prob.log_likelyhood

        trend_model = ExternalDrift(krg_lo, w_trend, shared_trend=self.shared_trend)

        info.update(
            noise=noise.item() if ydim_hi == 1 else noise,
            log_likelyhood=log_likelyhood.item() if ydim_hi == 1 else log_likelyhood,
            trend_model=trend_model.name,
        )

        krg_mf = KernelSurrogateModel(_x_hi, w_kernel, kernel_func,
                                      trend=trend_model, info=info, hparam=_hparam,
                                      xoffset=_xoffset, xscale=_xscale,
                                      yoffset=_yoffset, yscale=_yscale)

        return krg_mf


class MFKrigingOptim(KrigingOptim):

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 kernel_func: PosDefKernel,
                 trend_mtx: np.ndarray,
                 shared_trend: bool = True,
                 regularize: bool = False,
                 maxiter: int = 500,
                 optim_restarts: int = 0):

        self.shared_trend = bool(shared_trend)
        super().__init__(x, y, kernel_func, trend_mtx,
                         gls_trend=True, regularize=regularize,
                         maxiter=maxiter, optim_restarts=optim_restarts)

    def _trend_init(self, t_mtx: np.ndarray) -> None:

        if self.shared_trend:
            self.tdim = t_mtx.shape[1]
            self.trend_mtx = t_mtx
            self.w_trend = np.zeros((self.tdim, self.ydim))

        else:
            self.tdim = 1
            self.trend_mtx = t_mtx
            self.w_trend = np.zeros((1, self.ydim))

        self._w_t = self.w_trend
        self._t_mtx = self.trend_mtx

    def set_y_index(self, idx: [None, int]) -> None:
        super().set_y_index(idx)

        if self.idx is Ellipsis or self.shared_trend:
            self._t_mtx = self.trend_mtx
        else:
            self._t_mtx = self.trend_mtx[:, self.idx, None]


class ExternalDrift(BaseSurrogateModel):

    def __init__(self,
                 model: BaseSurrogateModel,
                 weights: np.ndarray,
                 shared_trend: bool = True):
        super().__init__()

        _weights = np.ascontiguousarray(weights)
        tdim, ydim = _weights.shape

        assert isinstance(model, BaseSurrogateModel)

        if shared_trend:
            assert model.ydim == tdim
        else:
            assert tdim == 1 and model.ydim == ydim

        self.model = model          # type: BaseSurrogateModel
        self.weights = _weights     # type: np.ndarray

        self.shared_trend = bool(shared_trend)  # type: bool

        self.xdim = model.xdim  # type: int
        self.ydim = ydim        # type: int
        self.info = model.info  # type: dict

    def _eval(self, x: np.ndarray,
              grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:

        y_out = self.model._eval(x)
        if self.shared_trend:
            y_out = np.dot(y_out, self.weights)
        else:
            y_out *= self.weights

        return y_out

    def _grad(self, x: np.ndarray) -> np.ndarray:

        dy_out = self.model._grad(x)
        if self.shared_trend:
            dy_out = np.dot(dy_out, self.weights)
        else:
            dy_out *= self.weights[None, ...]

        return dy_out
