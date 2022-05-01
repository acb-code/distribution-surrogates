import numpy as np

from abc import abstractmethod, ABC
from scipy.optimize import minimize, OptimizeResult

from ..common import BaseModel, BaseBuilder


class BaseSurrogateModel(BaseModel, ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.xdim = 1   # type: int
        self.ydim = 1   # type: int

        self.xoffset = None     # type: [None, np.ndarray]
        self.yoffset = None     # type: [None, np.ndarray]
        self.xscale = None      # type: [None, np.ndarray]
        self.yscale = None      # type: [None, np.ndarray]

        self.info = {}      # type: dict

    def __call__(self, x: np.ndarray,
                 grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:
        return self.eval(x, grad=grad)

    def eval(self, x: np.ndarray,
             grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:

        _x = self._preprocess(x, self.xdim, self.xoffset, self.xscale)

        if grad:
            y, dy = self._eval(_x, grad)
            y = self._postprocess(y, self.yoffset, self.yscale)
            dy = self._postprocess_dy(dy)
            return y, dy
        else:
            y = self._eval(_x)
            return self._postprocess(y, self.yoffset, self.yscale)

    def grad(self, x: np.ndarray) -> np.ndarray:

        _x = self._preprocess(x, self.xdim, self.xoffset, self.xscale)
        _y = self._grad(_x)

        return self._postprocess_dy(_y)

    def test_model(self, x: np.ndarray, y: np.ndarray, normalize: bool = True) -> np.ndarray:

        _y = self._check_array(y, self.ydim, copy=False)
        npts = _y.shape[0]

        error = np.linalg.norm(_y - self.eval(x).reshape(-1, self.ydim), axis=0) / npts**0.5

        if normalize:
            assert npts > 1
            error /= np.std(_y, axis=0)

        return error

    @abstractmethod
    def _eval(self, x: np.ndarray,
              grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:
        pass

    @abstractmethod
    def _grad(self, x: np.ndarray) -> np.ndarray:
        pass

    def _set_normalize(self,
                       xoffset: [None, np.ndarray],
                       xscale: [None, np.ndarray],
                       yoffset: [None, np.ndarray],
                       yscale: [None, np.ndarray]):

        if xoffset is not None:
            _xoffset = np.asarray(xoffset).ravel()
            assert _xoffset.size == self.xdim
            self.xoffset = _xoffset

        if xscale is not None:
            _xscale = np.asarray(xscale).ravel()
            assert _xscale.size == self.xdim
            self.xscale = _xscale

        if yoffset is not None:
            _ymean = np.asarray(yoffset).ravel()
            assert _ymean.size == self.ydim
            self.yoffset = _ymean

        if yscale is not None:
            _yscale = np.asarray(yscale).ravel()
            assert _yscale.size == self.ydim
            self.yscale = _yscale

    def _postprocess_dy(self, dy: np.ndarray) -> [float, np.ndarray]:

        if self.yscale is not None:
            dy *= self.yscale[None, None, :]
        if self.xscale is not None:
            dy /= self.xscale[None, :, None]

        if dy.size == 1:
            return dy.item()
        else:
            return dy.squeeze()


class BaseSurrogateBuilder(BaseBuilder, ABC):

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = bool(normalize)

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> BaseSurrogateModel:
        pass


class BaseMFSurrogateBuilder(BaseBuilder, ABC):

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = bool(normalize)

    @abstractmethod
    def train(self,
              x_hi: np.ndarray, y_hi: np.ndarray,
              x_lo: np.ndarray, y_lo: np.ndarray,
              **kwargs) -> BaseSurrogateModel:
        pass


class BaseSurrogateOptim(ABC):

    def __init__(self,
                 maxiter: int = 500,
                 optim_restarts: int = 0):

        self.xdim = 1   # type: int
        self.ydim = 1   # type: int

        self.maxiter = int(maxiter)                 # type: int
        self.optim_restarts = int(optim_restarts)   # type: int

        self.dvar = None    # type: [None, np.ndarray]
        self.obj = None     # type: [None, float]
        self.dobj = None    # type: [None, np.ndarray]

        self.idx = Ellipsis     # type: [Ellipsis, int]

    def set_y_index(self, idx: [None, int]) -> None:

        if idx is None:
            _idx = Ellipsis
        else:
            _idx = int(idx)

        self.idx = _idx

        self.dvar = None    # type: [None, np.ndarray]
        self.obj = None     # type: [None, float]
        self.dobj = None    # type: [None, np.ndarray]

    @property
    @abstractmethod
    def x0(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        pass

    @abstractmethod
    def eval(self, dvar: np.ndarray) -> float:
        pass

    def grad(self, dvar: np.ndarray) -> np.ndarray:
        pass

    def optimize(self) -> list:

        opt_results = []

        for i in range(self.ydim):

            self.set_y_index(i)
            opt_bounds = self.bounds
            opt_yi = OptimizeResult(fun=np.inf, x=-np.inf)

            for ii in range(self.optim_restarts + 1):

                if ii == 0:
                    opt_x0 = self.x0
                else:
                    opt_x0 = np.random.uniform(opt_bounds[:, 0], opt_bounds[:, 1])

                opt_tmp = minimize(self.eval, opt_x0,
                                   bounds=opt_bounds,
                                   jac=self.grad,
                                   method='SLSQP',
                                   options={'maxiter': self.maxiter})

                if opt_tmp.fun < opt_yi.fun or ii == 0:
                    opt_yi.update(opt_tmp)

            opt_results.append(opt_yi.copy())

            # re-run optimum
            self.eval(opt_yi.x)

        # reset y index
        self.set_y_index(None)

        return opt_results
