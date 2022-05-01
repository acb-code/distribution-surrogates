import numpy as np

from abc import ABC, abstractmethod
from scipy.optimize import minimize, OptimizeResult
from .common import BaseSurrogateModel, BaseSurrogateBuilder, BaseMFSurrogateBuilder
from . import kernel_functions as k_f


class KernelSurrogateModel(BaseSurrogateModel):

    def __init__(self,
                 xcenter: np.ndarray,
                 weights: np.ndarray,
                 kernel: k_f.BaseKernel,
                 hparam: np.ndarray = None,
                 trend: BaseSurrogateModel = None,
                 info: dict = None,
                 xoffset: np.ndarray = None,
                 xscale: np.ndarray = None,
                 yoffset: np.ndarray = None,
                 yscale: np.ndarray = None):

        super().__init__()

        assert isinstance(kernel, k_f.BaseKernel)

        _xcenter = np.asarray(xcenter)
        _weigths = np.asarray(weights)

        assert _xcenter.ndim == 2 and _weigths.ndim == 2
        assert _xcenter.shape[0] == _weigths.shape[0]

        npts, xdim = _xcenter.shape
        ydim = _weigths.shape[1]

        if hparam is not None:
            assert kernel.has_hparam
            _hparam = np.asarray(hparam)

            if _hparam.ndim == 1:
                kernel.check_hparam(_hparam, xdim)

            else:
                assert _hparam.ndim == 2 and _hparam.shape[1] == ydim
                for i in range(ydim):
                    kernel.check_hparam(_hparam[:, i], xdim)
                if ydim == 1:
                    _hparam = _hparam.ravel()

        else:
            _hparam = None

        if trend is not None:
            assert isinstance(trend, BaseSurrogateModel)
            assert trend.xdim == xdim and trend.ydim == ydim

        if info is not None:
            assert isinstance(info, dict)
            self.info.update(info)

        self.xcenter = _xcenter     # type: np.ndarray
        self.weights = _weigths     # type: np.ndarray
        self.kernel = kernel        # type: k_f.BaseKernel
        self.hparam = _hparam       # type: np.ndarray

        self.npts = npts    # type: int
        self.xdim = xdim    # type: int
        self.ydim = ydim    # type: int

        self.trend = trend  # type: BaseSurrogateModel

        self._set_normalize(xoffset, xscale, yoffset, yscale)

    def eval_trend(self, x: np.ndarray,
                   grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:

        _x = self._preprocess(x, self.xdim, self.xoffset, self.xscale)
        y = self.trend._eval(_x, grad)

        return self._postprocess(y, self.yoffset, self.yscale)

    def _eval(self, x: np.ndarray,
              grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:

        if self.hparam is None or self.hparam.ndim == 1 or self.ydim == 1:
            k_out = self.kernel.eval(x, self.xcenter, hparam=self.hparam)
            y_out = np.dot(k_out, self.weights)

        else:
            y_out = np.empty((x.shape[0], self.ydim))
            for i in range(self.ydim):
                k_out = self.kernel.eval(x, self.xcenter, hparam=self.hparam[:, i])
                y_out[:, i] = np.dot(k_out, self.weights[:, i])

        if self.trend is not None:
            y_out += self.trend._eval(x)

        if grad:
            return y_out, self._grad(x)
        else:
            return y_out

    def _grad(self, x: np.ndarray) -> np.ndarray:

        if self.hparam is None or self.hparam.ndim == 1 or self.ydim == 1:
            dk_out = self.kernel.grad(x, self.xcenter, hparam=self.hparam)
            dy_out = np.dot(dk_out, self.weights)

        else:
            dy_out = np.empty((x.shape[0], self.xdim, self.ydim))
            for i in range(self.ydim):
                dk_out = self.kernel.grad(x, self.xcenter, hparam=self.hparam[:, i])
                dy_out[..., i] = np.dot(dk_out, self.weights[:, i])

        if self.trend is not None:
            dy_out += self.trend._grad(x)

        return dy_out


class KernelSurrogateBuilder(BaseSurrogateBuilder, ABC):

    _kernel_options = []

    def __init__(self,
                 kernel: str = '',
                 optimize: bool = True,
                 regularize: bool = False,
                 normalize: bool = True,
                 maxiter: int = 500,
                 optim_restarts: int = 0):

        super().__init__(normalize=normalize)

        assert float(maxiter).is_integer() and maxiter > 0
        assert float(optim_restarts).is_integer() and optim_restarts >= 0

        _kernel = str(kernel).lower()
        assert _kernel in self._kernel_options

        self.kernel = _kernel   # type: str

        self.optimize = bool(optimize)              # type: bool
        self.regularize = bool(regularize)          # type: bool
        self.maxiter = int(maxiter)                 # type: int
        self.optim_restarts = int(optim_restarts)   # type: int

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray,
              hparam: np.ndarray = None) -> KernelSurrogateModel:
        pass

    def _get_kernel_func(self) -> k_f.BaseKernel:
        return k_f.get_kernel_function(self.kernel)


class KernelMFSurrogateBuilder(BaseMFSurrogateBuilder, ABC):

    _kernel_options = []

    def __init__(self,
                 kernel: str = '',
                 optimize: bool = True,
                 regularize: bool = False,
                 normalize: bool = True,
                 maxiter: int = 500,
                 optim_restarts: int = 0):

        super().__init__(normalize=normalize)

        assert float(maxiter).is_integer() and maxiter > 0
        assert float(optim_restarts).is_integer() and optim_restarts >= 0

        _kernel = str(kernel).lower()
        assert _kernel in self._kernel_options

        self.kernel = _kernel   # type: str

        self.optimize = bool(optimize)              # type: bool
        self.regularize = bool(regularize)          # type: bool
        self.maxiter = int(maxiter)                 # type: int
        self.optim_restarts = int(optim_restarts)   # type: int

    @abstractmethod
    def train(self,
              x_hi: np.ndarray, y_hi: np.ndarray,
              x_lo: np.ndarray, y_lo: np.ndarray,
              hparam_hi: np.ndarray = None,
              hparam_lo: np.ndarray = None) -> KernelSurrogateModel:
        pass

    def _get_kernel_func(self) -> k_f.BaseKernel:
        return k_f.get_kernel_function(self.kernel)
