import numpy as np

from ..common import BaseModel, BaseBuilder
from ..dr.common import BaseDRModel, BaseDRBuilder, BaseS2DRBuilder
from ..surrogate.common import BaseSurrogateModel, BaseSurrogateBuilder, BaseMFSurrogateBuilder

from ..dr import (PCABuilder,
                  LPPBuilder,
                  ProcrustesBuilder,
                  ExtendedPODBuilder,
                  CommonPODBuilder)
from ..surrogate import (RBFBuilder,
                         KrigingBuilder,
                         BridgeFunctionBuilder,
                         MFKrigingBuilder,
                         PolynomialBuilder)


class GenericROM(BaseModel):

    def __init__(self, sm_model, dr_model):
        super().__init__()

        assert isinstance(sm_model, BaseSurrogateModel)
        assert isinstance(dr_model, BaseDRModel)

        assert sm_model.ydim == dr_model.zdim

        self.sm_model = sm_model
        self.dr_model = dr_model

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.eval(x)

    @property
    def xdim(self) -> int:
        return self.sm_model.xdim

    @property
    def ydim(self) -> int:
        return self.dr_model.xdim

    @property
    def zdim(self) -> int:
        return self.dr_model.zdim

    def eval(self, x: np.ndarray) -> np.ndarray:

        z = self.sm_model.eval(x)
        y = self.dr_model.expand(z)

        return y

    def test_model(self, x: np.ndarray, y: np.ndarray, normalize: bool = True) -> (float, float, float):

        _y = self._check_array(y, self.ydim, copy=False)
        npts = _y.shape[0]

        # total error
        # total_error = np.linalg.norm(_y - self.eval(x).reshape(-1, self.ydim), axis=1).mean()
        total_error = np.linalg.norm(_y - self.eval(x).reshape(-1, self.ydim)) / np.sqrt(npts)

        # reconstruction error
        recon_error = self.dr_model.test_model(_y, normalize=False)

        # regression error
        regr_error = np.sqrt(total_error**2 - recon_error**2)

        if normalize:
            # error_norm = np.linalg.norm(_y - _y.mean(axis=0), axis=1).mean()
            error_norm = np.linalg.norm(_y - _y.mean(axis=0)) / np.sqrt(npts)

            total_error /= error_norm
            recon_error /= error_norm
            regr_error /= error_norm

        return total_error, recon_error, regr_error


class GenericROMBuilder(BaseBuilder):

    _available_sm = ['rbf', 'kriging', 'polynomial']
    _available_dr = ['pca', 'lpp']

    def __init__(self,
                 sm: str = 'rbf',
                 dr: str = 'pca',
                 sm_options: [None, dict] = None,
                 dr_options: [None, dict] = None):
        super().__init__()

        _sm = str(sm).lower()
        _dr = str(dr).lower()

        assert _sm in self._available_sm
        assert _dr in self._available_dr

        if sm_options is None:
            sm_options = {}

        if dr_options is None:
            dr_options = {}

        assert isinstance(sm_options, dict)
        assert isinstance(dr_options, dict)

        if _sm == 'rbf':
            self.sm_builder = RBFBuilder(**sm_options)  # type: BaseSurrogateBuilder
        elif _sm == 'kriging':
            self.sm_builder = KrigingBuilder(**sm_options)
        elif _sm == 'polynomial':
            self.sm_builder = PolynomialBuilder(**sm_options)
        else:
            raise NotImplementedError

        if _dr == 'pca':
            self.dr_builder = PCABuilder(**dr_options)  # type: BaseDRBuilder
        elif _dr == 'lpp':
            self.dr_builder = LPPBuilder(**dr_options)
        else:
            raise NotImplementedError

    def train(self, x: np.ndarray, y: np.ndarray) -> GenericROM:

        _x, _y = self._check_data(x, y)

        z, dr_model = self.dr_builder.train(y, model=True)
        sm_model = self.sm_builder.train(x, z)

        rom = GenericROM(sm_model, dr_model)

        return rom


class GenericMFROMBuilder(BaseBuilder):

    _available_sm = ['bridge_function', 'kriging']
    _available_dr = ['procrustes', 'extended-pod', 'common-pod']

    def __init__(self,
                 sm: str = 'bridge_function',
                 dr: str = 'procrustes',
                 sm_options: [None, dict] = None,
                 dr_options: [None, dict] = None):
        super().__init__()

        _sm = str(sm).lower()
        _dr = str(dr).lower()

        assert _sm in self._available_sm
        assert _dr in self._available_dr

        if sm_options is None:
            _sm_options = {}
        else:
            assert isinstance(sm_options, dict)
            _sm_options = sm_options

        if dr_options is None:
            _dr_options = {}
        else:
            assert isinstance(dr_options, dict)
            _dr_options = dr_options

        if _dr == 'procrustes':
            self.dr_builder = ProcrustesBuilder(**_dr_options)   # type: BaseS2DRBuilder
        elif _dr == 'extended-pod':
            self.dr_builder = ExtendedPODBuilder(**_dr_options)
        elif _dr == 'common-pod':
            self.dr_builder = CommonPODBuilder(**_dr_options)
        else:
            raise NotImplementedError

        if _sm == 'bridge_function':
            self.sm_builder = BridgeFunctionBuilder(**_sm_options)   # type: BaseMFSurrogateBuilder
        elif _sm == 'kriging':
            _sm_options.update(shared_trend=False)
            self.sm_builder = MFKrigingBuilder(**_sm_options)
        else:
            raise NotImplementedError

    def train(self,
              x_hi: np.ndarray, y_hi: np.ndarray,
              x_lo: np.ndarray, y_lo: np.ndarray) -> GenericROM:

        _x_hi, _y_hi = self._check_data(x_hi, y_hi)
        _x_lo, _y_lo = self._check_data(x_lo, y_lo)

        assert _x_hi.shape[0] <= _x_lo.shape[0]

        z_hi, z_lo, dr_model_hi, _ = self.dr_builder.train(y_hi, y_lo, model=True)

        sm_model = self.sm_builder.train(_x_hi, z_hi, _x_lo, z_lo)

        rom = GenericROM(sm_model, dr_model_hi)

        return rom
