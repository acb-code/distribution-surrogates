import os
import pickle
import numpy as np

from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self):
        self.dtype = np.float64

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _check_array(self, x: np.ndarray, xdim: int, copy=False) -> np.ndarray:

        _x = np.array(x, dtype=self.dtype, copy=copy, ndmin=2)

        if _x.ndim > 2:
            raise ValueError('Array dimensions greater than 2')
        elif xdim == 1:
            _x = _x.reshape(_x.size, 1)
        else:
            assert _x.shape[1] == xdim

        return _x

    @abstractmethod
    def test_model(self, *args, **kwargs):
        pass

    def _preprocess(self, x: np.ndarray, dim: int, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:

        is_copy = False if (offset is None and scale is None) else True

        _x = self._check_array(x, dim, copy=is_copy)

        if offset is not None:
            _x -= offset
        if scale is not None:
            _x /= scale

        return _x

    def _postprocess(self, x: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> [float, np.ndarray]:

        if scale is not None:
            x *= scale
        if offset is not None:
            x += offset

        if x.size == 1:
            return x.item()
        else:
            return x.squeeze()

    def save(self, filename: str, overwrite: bool = True) -> None:

        if os.path.isfile(filename) and not overwrite:
            raise FileExistsError

        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)


class BaseBuilder(ABC):

    def __init__(self):
        self.dtype = np.float64

    @abstractmethod
    def train(self, *args, **kwargs) -> BaseModel:
        pass

    def _check_data(self, *data: np.ndarray, copy=False) -> [np.ndarray, tuple]:

        _data = []
        npts = 0

        for x in data:
            _x = np.array(x, dtype=self.dtype, copy=copy, ndmin=2)

            if _x.ndim > 2:
                raise ValueError('Array dimensions greater than 2')

            elif 1 in _x.shape:
                _x = _x.reshape(-1, 1)

            if not npts:
                npts = _x.shape[0]
            else:
                if _x.shape[0] != npts:
                    raise ValueError('Supplied arrays have inconsistent sizes')

            _data.append(_x)

        if len(_data) == 1:
            return _data[0]
        else:
            return tuple(_data)

    @staticmethod
    def _normalize_data(data: np.ndarray) -> (np.ndarray, float, float):

        mean = np.atleast_1d(np.mean(data, axis=0))
        std = np.atleast_1d(np.std(data, axis=0))
        std[std == 0.] = 1.

        data -= mean
        data /= std

        return data, mean, std

    @staticmethod
    def load(filename: str) -> BaseModel:

        if not os.path.isfile(filename):
            raise FileNotFoundError

        with open(filename, 'rb') as fid:
            model = pickle.load(fid)

        assert isinstance(model, BaseModel)

        return model
