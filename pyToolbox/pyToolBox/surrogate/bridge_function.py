import numpy as np

from .common import BaseSurrogateModel, BaseMFSurrogateBuilder, BaseSurrogateBuilder
from ..surrogate import RBFBuilder, PolynomialBuilder, KrigingBuilder


class BridgeFunctionBuilder(BaseMFSurrogateBuilder):

    _available_sm = ['rbf', 'kriging', 'polynomial']

    def __init__(self,
                 trend_sm: str = 'rbf',
                 bridge_sm: [None, str] = None,
                 trend_options: [None, dict] = None,
                 bridge_options: [None, dict] = None,
                 normalize: bool = True):

        super().__init__(normalize=normalize)

        _trend_sm = str(trend_sm).lower()

        assert _trend_sm in self._available_sm

        if trend_options is None:
            _trend_options = {'normalize': False}
        else:
            assert isinstance(trend_options, dict)
            _trend_options = trend_options.copy()
            _trend_options.update(normalize=False)

        if _trend_sm == 'rbf':
            self.trend_builder = RBFBuilder(**_trend_options)    # type: BaseSurrogateBuilder
        elif _trend_sm == 'kriging':
            self.trend_builder = KrigingBuilder(**_trend_options)
        elif _trend_sm == 'polynomial':
            self.trend_builder = PolynomialBuilder(**_trend_options)
        else:
            raise NotImplementedError

        if bridge_sm is None:
            self.bridge_builder = self.trend_builder            # type: BaseSurrogateBuilder
        else:
            _bridge_sm = str(bridge_sm).lower()

            assert _bridge_sm in self._available_sm

            if bridge_options is None:
                _bridge_options = _trend_options
            else:
                assert isinstance(bridge_options, dict)
                _bridge_options = bridge_options.copy()
                _bridge_options.update(normalize=False)

            if _bridge_sm == 'rbf':
                self.bridge_builder = RBFBuilder(**_bridge_options)
            elif _bridge_sm == 'kriging':
                self.bridge_builder = KrigingBuilder(**_bridge_options)
            elif _bridge_sm == 'polynomial':
                self.bridge_builder = PolynomialBuilder(**_bridge_options)
            else:
                raise NotImplementedError

    def train(self,
              x_hi: np.ndarray, y_hi: np.ndarray,
              x_lo: np.ndarray, y_lo: np.ndarray,
              hparam_hi=None, hparam_lo=None) -> BaseSurrogateModel:

        _x_hi, _y_hi = self._check_data(x_hi, y_hi, copy=True)
        _x_lo, _y_lo = self._check_data(x_lo, y_lo, copy=True)

        npts_hi, xdim = _x_hi.shape
        npts_lo = _x_lo.shape[0]
        ydim = _y_hi.shape[1]

        assert 1 <= npts_hi <= npts_lo
        assert xdim == _x_lo.shape[1]
        assert ydim == _y_lo.shape[1]

        if self.normalize:
            _x_hi, _xoffset, _xscale = self._normalize_data(_x_hi)
            _x_lo -= _xoffset
            _x_lo /= _xscale

            _yoffset = _yscale = None

        else:
            _xoffset = _xscale = None
            _yoffset = _yscale = None

        npts_lo, xdim = _x_lo.shape
        npts_hi, ydim = _y_hi.shape

        assert npts_hi <= npts_lo

        trend_sm = self.trend_builder.train(_x_lo, y_lo, hparam=hparam_lo)
        hi_lo_diff = _y_hi - trend_sm.eval(_x_hi).reshape(npts_hi, ydim)
        bridge_sm = self.bridge_builder.train(_x_hi, hi_lo_diff, hparam=hparam_hi)

        bridge_sm = BridgeFunction(trend_sm, bridge_sm,
                                   xoffset=_xoffset, xscale=_xscale,
                                   yoffset=_yoffset, yscale=_yscale)

        return bridge_sm


class BridgeFunction(BaseSurrogateModel):

    def __init__(self, trend_sm: BaseSurrogateModel, bridge_sm: BaseSurrogateModel,
                 xoffset: [None, np.ndarray] = None, xscale: [None, np.ndarray] = None,
                 yoffset: [None, np.ndarray] = None, yscale: [None, np.ndarray] = None):

        super().__init__()

        assert isinstance(bridge_sm, BaseSurrogateModel)
        assert isinstance(trend_sm, BaseSurrogateModel)

        self.xdim = trend_sm.xdim   # type: int
        self.ydim = trend_sm.ydim   # type: int

        self.trend_sm = trend_sm    # type: BaseSurrogateModel
        self.bridge_sm = bridge_sm  # type: BaseSurrogateModel

        self.info.update(trend_info=trend_sm.info,
                         bridge_info=bridge_sm.info)

        self._set_normalize(xoffset, xscale, yoffset, yscale)

    def _eval(self, x: np.ndarray,
              grad: bool = False) -> [np.ndarray, (np.ndarray, np.ndarray)]:

        y_out = self.trend_sm._eval(x)
        y_out += self.bridge_sm._eval(x)

        if grad:
            return y_out, self._grad(x)
        else:
            return y_out

    def _grad(self, x: np.ndarray) -> np.ndarray:

        dy_out = self.trend_sm._grad(x)
        dy_out += self.bridge_sm._grad(x)

        return dy_out
