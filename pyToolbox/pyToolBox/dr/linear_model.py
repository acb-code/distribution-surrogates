import numpy as np
import scipy.linalg as la

from abc import ABC, abstractmethod
from .common import BaseDRModel, BaseDRBuilder, BaseS2DRBuilder


class LinearDRModel(BaseDRModel):

    def __init__(self,
                 basis: np.ndarray,
                 inv_basis: np.ndarray = None,
                 orthonorm: bool = False,
                 info: dict = None,
                 xoffset: np.ndarray = None,
                 xscale: np.ndarray = None,
                 zoffset: np.ndarray = None,
                 zscale: np.ndarray = None):
        super().__init__()

        _basis = np.ascontiguousarray(basis)
        assert _basis.ndim == 2
        self.basis = _basis     # type: np.ndarray

        if orthonorm:
            self.inv_basis = self.basis.T
        elif inv_basis is not None:
            _inv_basis = np.asarray(inv_basis)
            assert _inv_basis.T.shape == self.basis.shape
            self.inv_basis = _inv_basis
        else:
            self.inv_basis = None

        self.info = {}          # type: dict
        if info is not None:
            self.info.update(info)

        if xoffset is not None:
            _xoffset = np.asarray(xoffset).ravel()
            assert _xoffset.size == self.xdim or _xoffset.size == 1
            self.xoffset = _xoffset     # type: np.ndarray
        else:
            self.xoffset = None

        if xscale is not None:
            _xscale = np.asarray(xscale).ravel()
            assert _xscale.size == self.xdim or _xscale.size == 1
            self.xscale = _xscale   # type: np.ndarray
        else:
            self.xscale = None

        if zoffset is not None:
            _zoffset = np.asarray(zoffset).ravel()
            assert _zoffset.size == self.zdim or _zoffset.size == 1
            self.zoffset = _zoffset     # type: np.ndarray
        else:
            self.zoffset = None

        if zscale is not None:
            _zscale = np.asarray(zscale).ravel()
            assert _zscale.size == self.zdim or _zscale.size == 1
            self.zscale = _zscale   # type: np.ndarray
        else:
            self.zscale = None

    @property
    def is_orthonormal(self):
        if self.inv_basis is not None:
            return self.basis is self.inv_basis.base
        else:
            return False

    @property
    def has_inverse(self):
        return self.inv_basis is not None

    @property
    def xdim(self):
        return self.basis.shape[1]

    @property
    def zdim(self):
        return self.basis.shape[0]

    def compress(self, x):

        if not self.has_inverse:
            raise ValueError

        _x = self._preprocess(x, self.xdim, self.xoffset, self.xscale)
        z = np.dot(_x, self.inv_basis)
        z = self._postprocess(z, self.zoffset, self.zscale)

        return z

    def expand(self, z):

        _z = self._preprocess(z, self.zdim, self.zoffset, self.zscale)
        x = np.dot(_z, self.basis)
        x = self._postprocess(x, self.xoffset, self.xscale)

        return x

    def test_model(self, x: np.ndarray, normalize: bool = True) -> float:

        _x = self._preprocess(x, self.xdim, self.xoffset, self.xscale)
        npts = _x.shape[0]

        if self.has_inverse:
            _basis = self.basis
            _inv_basis = self.inv_basis
        else:
            _, _basis = la.rq(self.basis, mode='economic')
            _inv_basis = _basis.T

        # error = la.norm(np.dot(np.dot(_x, _inv_basis), _basis) - _x, axis=1).mean()
        error = la.norm(np.dot(np.dot(_x, _inv_basis), _basis) - _x) / np.sqrt(npts)

        if normalize:
            # error /= la.norm(_x - _x.mean(axis=0), axis=1).mean()
            error /= la.norm(_x - _x.mean(axis=0)) / np.sqrt(npts)

        return error


class LinearDRBuilder(BaseDRBuilder, ABC):

    @abstractmethod
    def train(self, data: np.ndarray, model: bool = True)\
            -> [np.ndarray, (np.ndarray, LinearDRModel)]:
        pass


class LinearS2DRBuilder(BaseS2DRBuilder, ABC):

    @abstractmethod
    def train(self, data1: np.ndarray, data2: np.ndarray, model: bool = True)\
            -> [(np.ndarray, np.ndarray),
                (np.ndarray, np.ndarray, BaseDRModel, BaseDRModel)]:
        pass
