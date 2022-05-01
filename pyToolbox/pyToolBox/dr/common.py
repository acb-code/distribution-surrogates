import numpy as np

from abc import ABC, abstractmethod
from ..common import BaseModel, BaseBuilder


class BaseDRModel(BaseModel, ABC):

    @property
    def has_inverse(self) -> bool:
        return False

    @property
    @abstractmethod
    def xdim(self) -> int:
        pass

    @property
    @abstractmethod
    def zdim(self) -> int:
        pass

    @abstractmethod
    def compress(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def expand(self, data: np.ndarray) -> np.ndarray:
        pass


class BaseDRBuilder(BaseBuilder, ABC):

    def __init__(self, ncomp: [None, int, float] = None):
        super().__init__()

        if ncomp is None:
            pass

        elif isinstance(ncomp, (int, np.integer)):
            assert ncomp > 0
            ncomp = int(ncomp)

        elif isinstance(ncomp, (float, np.floating)):
            assert 0 < ncomp <= 1
            ncomp = float(ncomp)

        else:
            raise ValueError

        self.ncomp = ncomp  # type: [None, int, float]

    @abstractmethod
    def train(self, *args, **kwargs) -> [np.ndarray,
                                         (np.ndarray, BaseDRModel)]:
        pass


class BaseS2DRBuilder(BaseBuilder, ABC):

    def __init__(self, ncomp: [None, int, float] = None):
        super().__init__()

        if ncomp is None:
            pass

        elif isinstance(ncomp, (int, np.integer)):
            assert ncomp > 0
            ncomp = int(ncomp)

        elif isinstance(ncomp, (float, np.floating)):
            assert 0 < ncomp <= 1
            ncomp = float(ncomp)

        else:
            raise ValueError

        self.ncomp = ncomp  # type: [None, int, float]

    @abstractmethod
    def train(self, *args, **kwargs) -> [(np.ndarray, np.ndarray),
                                         (np.ndarray, np.ndarray, BaseDRModel, BaseDRModel)]:
        pass
