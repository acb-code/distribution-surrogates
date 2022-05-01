import numpy as np
import scipy.linalg as la

from .linear_model import LinearS2DRBuilder, LinearDRModel, LinearDRBuilder


class ExtendedPODBuilder(LinearS2DRBuilder):

    def train(self, data1: np.ndarray, data2: np.ndarray, model: bool = True) -> \
            [(np.ndarray, np.ndarray), (np.ndarray, np.ndarray, LinearDRModel, LinearDRModel)]:

        x = self._check_data(data1, copy=True)
        y = self._check_data(data2, copy=True)
        _model = bool(model)

        xnpts, xndim = x.shape
        ynpts, yndim = y.shape

        assert xnpts <= ynpts and xndim == yndim

        mean = np.mean(x, axis=0)
        x -= mean
        y -= mean

        # noinspection PyTupleAssignmentBalance
        _, xsvals, xbasis = la.svd(x, full_matrices=True)
        xvar_vct = (xsvals ** 2.) / xnpts

        if self.ncomp is None or self.ncomp < 1:
            train_ric = 1.0 if self.ncomp is None else self.ncomp

            ric_vct = np.cumsum(xvar_vct)
            ric_vct /= ric_vct[-1]
            xncomp = np.searchsorted(ric_vct, train_ric) + 1

        else:
            xncomp = min(self.ncomp, xnpts)

        xbasis = xbasis[:xncomp]

        u = y - (y.dot(xbasis.T)).dot(xbasis)
        d = np.mean(u, axis=0) * ynpts / (xnpts + ynpts)
        u = np.append(u, d[None, :] * (xnpts**0.5 + 1), axis=0)
        u -= d

        mean += d
        x -= d
        y -= d

        # noinspection PyTupleAssignmentBalance
        _, usvals, ubasis = la.svd(u, full_matrices=False)
        uvar_vct = (usvals ** 2.) / ynpts

        if self.ncomp is None or self.ncomp < 1:
            train_ric = 1.0 if self.ncomp is None else self.ncomp

            yvar_total = la.norm(y) ** 2 / ynpts
            uvar_total = uvar_vct.sum()

            if 1 - uvar_total/yvar_total >= train_ric:
                uncomp = 0

            else:
                uvar_vct[0] += yvar_total
                ric_vct = np.cumsum(uvar_vct)
                ric_vct /= ric_vct[-1]
                uncomp = np.searchsorted(ric_vct, train_ric) + 1

        else:
            uncomp = min(self.ncomp, ynpts) - xncomp

        if uncomp > 0:
            ubasis = ubasis[:uncomp]
            basis = np.vstack([xbasis, ubasis])

        else:
            basis = np.ascontiguousarray(xbasis)

        del ubasis, usvals, xbasis, xsvals

        z1 = x.dot(basis.T)
        z2 = y.dot(basis.T)

        if model:

            info = {
                'model': 'ExtendedPOD',
            }

            model = LinearDRModel(basis, orthonorm=True, info=info, xoffset=mean)

            return z1, z2, model, model

        else:
            return z1, z2
    #
    # def train(self, data1: np.ndarray, data2: np.ndarray, model: bool = True) -> \
    #         [(np.ndarray, np.ndarray), (np.ndarray, np.ndarray, LinearDRModel, LinearDRModel)]:
    #
    #     x = self._check_data(data1, copy=True)
    #     y = self._check_data(data2, copy=True)
    #     _model = bool(model)
    #
    #     xnpts, xndim = x.shape
    #     ynpts, yndim = y.shape
    #
    #     assert xnpts <= ynpts and xndim == yndim
    #
    #     xmean = np.mean(x, axis=0)
    #     x -= xmean
    #     y -= xmean
    #
    #     # ymean = np.mean(y, axis=0)
    #     # y -= ymean
    #
    #     # noinspection PyTupleAssignmentBalance
    #     _, xsvals, xspace = la.svd(x, full_matrices=True)
    #     xvar_vct = (xsvals ** 2.) / xnpts
    #
    #     if self.ncomp is None or self.ncomp < 1:
    #         train_ric = 1.0 if self.ncomp is None else self.ncomp
    #
    #         ric_vct = np.cumsum(xvar_vct)
    #         ric_vct /= ric_vct[-1]
    #         xncomp = np.searchsorted(ric_vct, train_ric) + 1
    #
    #     else:
    #         xncomp = min(self.ncomp, xnpts)
    #
    #     xbasis, xnull = xspace[:xncomp], xspace[xncomp:]
    #
    #     yp = y - (y.dot(xbasis.T)).dot(xbasis)
    #     d = np.mean(yp, axis=0) * ynpts / (xnpts + ynpts)
    #     yp = np.append(yp, d[None, :] * (xnpts**0.5 + 1), axis=0)
    #     yp -= d
    #
    #     x -= d
    #     y -= d
    #
    #     u = yp.dot(xnull.T)
    #     del yp
    #     # noinspection PyTupleAssignmentBalance
    #     _, usvals, uspace = la.svd(u, full_matrices=False)
    #     uvar_vct = (usvals ** 2.) / ynpts
    #
    #     if self.ncomp is None or self.ncomp < 1:
    #         train_ric = 1.0 if self.ncomp is None else self.ncomp
    #
    #         yvar_total = la.norm(y) ** 2 / ynpts
    #         uvar_total = uvar_vct.sum()
    #
    #         if 1 - uvar_total/yvar_total >= train_ric:
    #             uncomp = 0
    #
    #         else:
    #             uvar_vct[0] += yvar_total
    #             ric_vct = np.cumsum(uvar_vct)
    #             ric_vct /= ric_vct[-1]
    #             uncomp = np.searchsorted(ric_vct, train_ric) + 1
    #
    #     else:
    #         uncomp = min(self.ncomp, ynpts) - xncomp
    #
    #     if uncomp > 0:
    #         ubasis = uspace[:uncomp]
    #         ybasis = ubasis.dot(xnull)
    #
    #         basis = np.vstack([xbasis, ybasis])
    #         del ubasis, uspace, usvals, ybasis, xbasis, xnull, xspace
    #
    #     else:
    #         basis = np.ascontiguousarray(xbasis)
    #         del uspace, usvals, xbasis, xnull, xspace
    #
    #     mean = xmean + d
    #
    #     z1 = x.dot(basis.T)
    #     z2 = y.dot(basis.T)
    #
    #     if model:
    #
    #         info = {
    #             'model': 'ExtendedPOD',
    #         }
    #
    #         model = LinearDRModel(basis, orthonorm=True, info=info, xoffset=mean)
    #
    #         return z1, z2, model, model
    #
    #     else:
    #         return z1, z2
