import numpy as np
import scipy.linalg as la

from .linear_model import LinearS2DRBuilder, LinearDRModel, LinearDRBuilder


class CommonPODBuilder(LinearS2DRBuilder):

    def train(self, data1: np.ndarray, data2: np.ndarray, model: bool = True) -> \
            [(np.ndarray, np.ndarray), (np.ndarray, np.ndarray, LinearDRModel, LinearDRModel)]:

        x = self._check_data(data1, copy=False)
        y = self._check_data(data2, copy=False)
        _model = bool(model)

        xnpts, xndim = x.shape
        ynpts, yndim = y.shape

        xynpts = xnpts + ynpts

        assert xnpts <= ynpts and xndim == yndim

        xy = np.vstack([x, y])
        xymean = np.mean(xy, axis=0)
        xy -= xymean

        # noinspection PyTupleAssignmentBalance
        z, xysvals, xybasis = la.svd(xy, full_matrices=True)
        xyvar_vct = (xysvals ** 2.) / xynpts
        xyvar_total = xyvar_vct.sum()

        if self.ncomp is None or self.ncomp < 1:
            train_ric = 1.0 if self.ncomp is None else self.ncomp

            ric_vct = np.cumsum(xyvar_vct)
            ric_vct /= ric_vct[-1]
            xyncomp = np.searchsorted(ric_vct, train_ric) + 1

        else:
            xyncomp = min(self.ncomp, xnpts)

        z = z[:, :xyncomp] * xysvals[None, :xyncomp]
        zx, zy = np.vsplit(z, [xnpts])

        if model:
            xyvar_vct = xyvar_vct[:xyncomp].copy()
            xyvar_sum = xyvar_vct.sum()
            train_ric = xyvar_sum / xyvar_total
            error = np.sqrt(max(xyvar_total - xyvar_sum, 0) / xndim)

            xybasis = xybasis[:xyncomp].copy()

            info = {
                'model': 'CommonPOD',
                'RIC': train_ric,
                'var': xyvar_vct,
                'error': error,
            }

            model = LinearDRModel(xybasis, orthonorm=True, info=info, xoffset=xymean)

            return zx, zy, model, model

        else:
            return zx, zy
