import numpy as np

from scipy.spatial.distance import cdist, pdist, squareform
from abc import ABC, abstractmethod

_TINY = np.finfo(np.float_).tiny


class BaseKernel(ABC):

    @property
    def name(self) -> str:
        return self.__class__.__name__[:-6]

    @property
    def has_hparam(self) -> bool:
        return isinstance(self, PosDefKernel)

    def get_default_hparam(self, xdim: int) -> [None, np.ndarray]:
        return None

    @staticmethod
    def check_hparam(hparam: np.ndarray, xdim: int) -> None:
        pass

    def __call__(self, x1: np.ndarray, x2: np.ndarray,
                 hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                    (np.ndarray, np.ndarray)]:
        self.eval(x1, x2, hparam=hparam, grad=grad)

    @abstractmethod
    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:
        pass

    @abstractmethod
    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:
        pass


class PosDefKernel(BaseKernel, ABC):

    @staticmethod
    @abstractmethod
    def optim_to_hyper(oparam: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def hyper_to_optim(hparam: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def check_hparam(hparam: np.ndarray, xdim: int) -> None:
        assert isinstance(hparam, np.ndarray)
        assert hparam.ndim == 1

    @abstractmethod
    def get_optim_bounds(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_optim_matrix(self, x: np.ndarray, oparam: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_optim_adjoint(self, x: np.ndarray, oparam: np.ndarray,
                          k_mtx: np.ndarray, k_mtx_b: np.ndarray) -> np.ndarray:
        pass


class LinearKernel(BaseKernel):

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        k_mtx = -cdist(x1, x2, 'euclidean')

        if grad:
            dk_mtx = (x1[..., None] - x2[..., None].T) / k_mtx[:, None, :]
            return k_mtx, dk_mtx
        else:
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        k_mtx = -cdist(x1, x2, 'euclidean')

        return (x1[..., None] - x2[..., None].T) / k_mtx[:, None, :]


class CubicKernel(BaseKernel):

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        k_mtx = cdist(x1, x2, 'euclidean') ** 3

        if grad:
            dk_mtx = 3 * k_mtx[:, None, :] * (x1[..., None] - x2[..., None].T)
            return k_mtx, dk_mtx
        else:
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        k_mtx = cdist(x1, x2, 'euclidean')

        return 3 * k_mtx[:, None, :] * (x1[..., None] - x2[..., None].T)


class TPSKernel(BaseKernel):

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        k_mtx = cdist(x1, x2, 'euclidean')
        tmp = np.log(k_mtx + _TINY)
        k_mtx[...] = k_mtx ** 2 * tmp
        if grad:
            dk_jac = (x1[..., None] - x2[..., None].T) * (tmp + 1)[:, None, :]
            return k_mtx, dk_jac
        else:
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        tmp = cdist(x1, x2, 'sqeuclidean')
        tmp = np.log(tmp + _TINY, out=tmp)

        return (x1[..., None] - x2[..., None].T) * (tmp + 1)[:, None, :]


class SqExpKernel(PosDefKernel):

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        hparam2 = hparam.item()**2
        k_mtx = cdist(x1, x2, 'sqeuclidean')
        k_mtx = np.exp(-0.5 * k_mtx / hparam2, out=k_mtx)
        if grad:
            dk_jac = (x2[..., None].T - x1[..., None]) * k_mtx[:, None, :] / hparam2
            return k_mtx, dk_jac
        else:
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        hparam2 = hparam.item() ** 2
        r2 = cdist(x1, x2, 'sqeuclidean')
        k_mtx = np.exp(-0.5 * r2 / hparam2)
        return (x2[..., None].T - x1[..., None]) * k_mtx[:, None, :] / hparam2

    @staticmethod
    def optim_to_hyper(oparam: np.ndarray) -> np.ndarray:
        return np.exp(oparam)

    @staticmethod
    def hyper_to_optim(hparam: np.ndarray) -> np.ndarray:
        return np.log(hparam + _TINY)

    @staticmethod
    def check_hparam(hparam: np.ndarray, xdim: int) -> None:
        PosDefKernel.check_hparam(hparam, xdim)
        assert hparam.size == 1
        assert np.all(hparam > 0)

    def get_default_hparam(self, xdim: int) -> np.ndarray:
        return np.array([0.5])

    def get_optim_bounds(self, x):

        r = pdist(x, 'euclidean')
        idx = np.nonzero(r)
        rmax = np.max(r[idx])
        rmin = np.min(r[idx])
        bounds = np.array([[rmin / 10, 10 * rmax]]) + _TINY

        return self.hyper_to_optim(bounds)

    def get_optim_matrix(self, x, oparam):

        hparam2 = np.exp(oparam.item())**2
        tmp = pdist(x, 'sqeuclidean')
        tmp *= -0.5 / hparam2
        tmp = np.exp(tmp, out=tmp)

        k_mtx = squareform(tmp)
        np.fill_diagonal(k_mtx, 1.)
        return k_mtx

    def get_optim_adjoint(self, x, oparam, k_mtx, k_mtx_b):

        triu_idx = np.triu_indices_from(k_mtx, 1)

        hparam2 = np.exp(oparam.item())**2
        d_k_mtx = pdist(x, 'sqeuclidean') / hparam2
        d_k_mtx *= k_mtx[triu_idx]
        d_k_mtx *= k_mtx_b[triu_idx]

        return 2. * d_k_mtx.sum()


class Matern32Kernel(SqExpKernel):

    _SQRT3 = np.sqrt(3)

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        sq3rh = -cdist(x1, x2, 'euclidean')
        sq3rh *= self._SQRT3 / hparam.item()

        k_mtx = np.exp(sq3rh)

        if grad:
            dk_jac = 3. * (x2[..., None].T - x1[..., None]) * k_mtx[:, None, :] / hparam**2
            k_mtx *= (1 - sq3rh)
            return k_mtx, dk_jac

        else:
            k_mtx *= (1 - sq3rh)
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        sq3rh = -cdist(x1, x2, 'euclidean')
        sq3rh *= self._SQRT3 / hparam.item()

        tmp = np.exp(sq3rh)

        return 3. * (x2[..., None].T - x1[..., None]) * tmp[:, None, :] / hparam**2

    def get_optim_matrix(self, x, oparam):

        hparam = np.exp(oparam.item())
        sq3rh = pdist(x, 'euclidean')
        sq3rh *= -self._SQRT3 / hparam

        k_mtx = np.exp(sq3rh)
        k_mtx *= (1. - sq3rh)

        k_mtx = squareform(k_mtx)
        np.fill_diagonal(k_mtx, 1.)
        return k_mtx

    def get_optim_adjoint(self, x, oparam, k_mtx, k_mtx_b):

        triu_idx = np.triu_indices_from(k_mtx, 1)

        hparam = np.exp(oparam.item())
        sq3r = pdist(x, 'euclidean')
        sq3r *= -self._SQRT3 / hparam

        d_k_mtx = np.exp(sq3r)
        d_k_mtx *= sq3r**2
        d_k_mtx *= k_mtx_b[triu_idx]

        return 2 * d_k_mtx.sum()


class Matern52Kernel(SqExpKernel):

    _SQRT5 = np.sqrt(5)

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        sq5rh = cdist(x1, x2, 'euclidean')
        sq5rh *= -self._SQRT5 / hparam.item()

        k_mtx = np.exp(sq5rh)

        if grad:
            dk_jac = 5. / 3. * k_mtx * (1. - sq5rh) / hparam ** 2
            dk_jac = (x2[..., None].T - x1[..., None]) * dk_jac[:, None, :]

            k_mtx *= (1 - sq5rh + sq5rh**2/3.)
            return k_mtx, dk_jac

        else:
            k_mtx *= (1 - sq5rh + sq5rh**2/3.)
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        sq5rh = cdist(x1, x2, 'euclidean')
        sq5rh *= -self._SQRT5 / hparam.item()

        tmp = np.exp(sq5rh)

        dk_jac = 5. / 3. * tmp * (1. - sq5rh) / hparam ** 2
        dk_jac = (x2[..., None].T - x1[..., None]) * dk_jac[:, None, :]

        return dk_jac

    def get_optim_matrix(self, x, oparam):

        hparam = np.exp(oparam.item())
        sq5rh = pdist(x, 'euclidean')
        sq5rh *= -self._SQRT5 / hparam

        k_mtx = np.exp(sq5rh)
        k_mtx *= (1. - sq5rh + sq5rh**2 / 3.)

        k_mtx = squareform(k_mtx)
        np.fill_diagonal(k_mtx, 1.)
        return k_mtx

    def get_optim_adjoint(self, x, oparam, k_mtx, k_mtx_b):

        triu_idx = np.triu_indices_from(k_mtx, 1)

        hparam = np.exp(oparam.item())
        sq5r = pdist(x, 'euclidean')
        sq5r *= -self._SQRT5 / hparam

        d_k_mtx = np.exp(sq5r)
        d_k_mtx *= sq5r**2
        d_k_mtx *= (1. - sq5r)
        d_k_mtx *= k_mtx_b[triu_idx]

        return 2./3. * d_k_mtx.sum()


class ARDSqExpKernel(PosDefKernel):

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        hparam2 = hparam**2
        k_mtx = -0.5 * cdist(x1, x2, 'seuclidean', V=hparam2)**2
        np.exp(k_mtx, out=k_mtx)
        if grad:
            dk_jac = (x2[..., None].T - x1[..., None]) * k_mtx[:, None, :] / hparam2[None, :, None]
            return k_mtx, dk_jac
        else:
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        hparam2 = hparam ** 2
        tmp = -0.5 * cdist(x1, x2, 'seuclidean', V=hparam2) ** 2
        np.exp(tmp, out=tmp)
        return (x2[..., None].T - x1[..., None]) * tmp[:, None, :] / hparam2[None, :, None]

    @staticmethod
    def optim_to_hyper(oparam: np.ndarray) -> np.ndarray:
        return np.exp(oparam)

    @staticmethod
    def hyper_to_optim(hparam: np.ndarray) -> np.ndarray:
        return np.log(hparam + _TINY)

    @staticmethod
    def check_hparam(hparam: np.ndarray, xdim: int) -> None:
        PosDefKernel.check_hparam(hparam, xdim)
        assert hparam.size == xdim
        assert np.all(hparam > 0)

    def get_default_hparam(self, xdim: int) -> np.ndarray:
        return np.full(xdim, 0.5)

    def get_optim_bounds(self, x):

        r = pdist(x, 'euclidean')
        idx = np.nonzero(r)
        rmax = np.max(r[idx])
        rmin = np.min(r[idx])
        bounds = np.array([[rmin / 10, 10 * rmax]]) + _TINY
        bounds = np.repeat(bounds, x.shape[1], axis=0)

        return self.hyper_to_optim(bounds)

    def get_optim_matrix(self, x, oparam):

        hparam2 = np.exp(oparam)**2
        tmp = -0.5 * pdist(x, 'seuclidean', V=hparam2)**2
        tmp = np.exp(tmp, out=tmp)

        k_mtx = squareform(tmp)
        np.fill_diagonal(k_mtx, 1.)
        return k_mtx

    def get_optim_adjoint(self, x, oparam, k_mtx, k_mtx_b):

        triu_idx = np.triu_indices_from(k_mtx, 1)
        triu_k_kb = k_mtx[triu_idx] * k_mtx_b[triu_idx]

        hparam2 = np.exp(oparam)**2
        d_oparam = np.empty_like(hparam2)

        for i, h in np.ndenumerate(hparam2):
            d_k_mtx = pdist(x[:, i], 'sqeuclidean') / h
            d_k_mtx *= triu_k_kb
            d_oparam[i] = 2. * d_k_mtx.sum()

        return d_oparam


class ARDMatern32Kernel(ARDSqExpKernel):

    _SQRT3 = np.sqrt(3)

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        sq3rh = -self._SQRT3 * cdist(x1, x2, 'seuclidean', V=hparam**2)
        k_mtx = np.exp(sq3rh)

        if grad:
            dk_jac = (3. * (x2[..., None].T - x1[..., None]) * k_mtx[:, None, :]
                      / hparam[None, :, None]**2)
            k_mtx *= (1 - sq3rh)
            return k_mtx, dk_jac

        else:
            k_mtx *= (1 - sq3rh)
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        tmp = -self._SQRT3 * cdist(x1, x2, 'seuclidean', V=hparam**2)
        np.exp(tmp, out=tmp)

        return 3. * (x2[..., None].T - x1[..., None]) * tmp[:, None, :] / hparam[None, :, None]**2

    def get_optim_matrix(self, x, oparam):

        hparam2 = np.exp(oparam)**2
        sq3rh = -self._SQRT3 * pdist(x, 'seuclidean', V=hparam2)

        k_mtx = np.exp(sq3rh)
        k_mtx *= (1. - sq3rh)

        k_mtx = squareform(k_mtx)
        np.fill_diagonal(k_mtx, 1.)
        return k_mtx

    def get_optim_adjoint(self, x, oparam, k_mtx, k_mtx_b):

        triu_idx = np.triu_indices_from(k_mtx, 1)

        hparam2 = np.exp(oparam)**2
        tmp = -self._SQRT3 * pdist(x, 'seuclidean', V=hparam2)
        np.exp(tmp, out=tmp)
        tmp *= k_mtx_b[triu_idx]

        d_oparam = np.empty_like(oparam)

        for i, h in np.ndenumerate(hparam2):
            d_k_mtx = pdist(x[:, i], 'sqeuclidean') / h
            d_k_mtx *= tmp

            d_oparam[i] = 6. * d_k_mtx.sum()

        return d_oparam


class ARDMatern52Kernel(ARDSqExpKernel):

    _SQRT5 = np.sqrt(5)

    def eval(self, x1: np.ndarray, x2: np.ndarray,
             hparam: np.ndarray = None, grad: bool = False) -> [np.ndarray,
                                                                (np.ndarray, np.ndarray)]:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        sq5rh = -self._SQRT5 * cdist(x1, x2, 'seuclidean', V=hparam**2)
        k_mtx = np.exp(sq5rh)

        if grad:
            dk_jac = 5. / 3. * k_mtx * (1. - sq5rh)
            dk_jac = (dk_jac[:, None, :] * (x2[..., None].T - x1[..., None])
                      / hparam[None, :, None] ** 2)

            k_mtx *= (1 - sq5rh + sq5rh ** 2 / 3.)
            return k_mtx, dk_jac

        else:
            k_mtx *= (1 - sq5rh + sq5rh ** 2 / 3.)
            return k_mtx

    def grad(self, x1: np.ndarray, x2: np.ndarray, hparam: np.ndarray = None) -> np.ndarray:

        if hparam is None:
            hparam = self.get_default_hparam(x1.shape[1])

        sq5rh = -self._SQRT5 * cdist(x1, x2, 'seuclidean', V=hparam ** 2)
        k_mtx = np.exp(sq5rh)

        dk_jac = 5. / 3. * k_mtx * (1. - sq5rh)
        dk_jac = (dk_jac[:, None, :] * (x2[..., None].T - x1[..., None])
                  / hparam[None, :, None] ** 2)

        return dk_jac

    def get_optim_matrix(self, x, oparam):

        hparam = np.exp(oparam)
        sq5rh = -self._SQRT5 * pdist(x, 'seuclidean', V=hparam**2)
        k_mtx = np.exp(sq5rh)
        k_mtx *= (1. - sq5rh + sq5rh ** 2 / 3.)

        k_mtx = squareform(k_mtx)
        np.fill_diagonal(k_mtx, 1.)
        return k_mtx

    def get_optim_adjoint(self, x, oparam, k_mtx, k_mtx_b):

        triu_idx = np.triu_indices_from(k_mtx, 1)

        hparam2 = np.exp(oparam)**2
        sq5r = -self._SQRT5 * pdist(x, 'seuclidean', V=hparam2)
        tmp = np.exp(sq5r)
        tmp *= (1. - sq5r)
        tmp *= k_mtx_b[triu_idx]

        d_oparam = np.empty_like(oparam)

        for i, h in np.ndenumerate(hparam2):
            d_k_mtx = pdist(x[:, i], 'sqeuclidean') / h
            d_k_mtx *= tmp

            d_oparam[i] = 10./3. * d_k_mtx.sum()

        return d_oparam


def get_kernel_function(name: str) -> BaseKernel:

    kernel_functions = {
        'linear': LinearKernel,
        'cubic': CubicKernel,
        'tps': TPSKernel,
        'sqexp': SqExpKernel,
        'matern32': Matern32Kernel,
        'matern52': Matern52Kernel,
        'ardsqexp': ARDSqExpKernel,
        'ardmatern32': ARDMatern32Kernel,
        'ardmatern52': ARDMatern52Kernel,
    }

    _name = str(name).lower()
    _name = _name.replace('_', '')

    assert _name in kernel_functions

    return kernel_functions[_name]()
