import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .misc import _compute_affinity_mtx
from .linear_model import LinearDRBuilder, LinearDRModel


class LPPBuilder(LinearDRBuilder):

    def __init__(self,
                 ncomp: [None, int] = None,
                 knn: int = 2,
                 weighted: bool = False,
                 sig: [None, float] = None,
                 rcond: float = None):

        super().__init__(ncomp)

        assert float(knn).is_integer() and knn >= 1

        if sig is None:
            _sig = None
        else:
            assert sig > 0.
            _sig = float(sig)

        if rcond is None:
            _rcond = None
        else:
            assert isinstance(rcond, (float, np.floating)) and rcond >= 0
            _rcond = float(rcond)

        self.knn = int(knn)
        self.weighted = bool(weighted)
        self.sig = _sig
        self.rcond = _rcond

    def train(self, data: np.ndarray, model: bool = True) \
            -> [np.ndarray, (np.ndarray, LinearDRModel)]:

        _data = self._check_data(data)
        _model = bool(model)

        if self.ncomp is not None:
            assert self.ncomp <= min(_data.shape)

        if _data.shape[0] > _data.shape[1]:
            pca_out = self._train_oversampled(_data, _model)
        else:
            pca_out = self._train_undersampled(_data, _model)

        return pca_out

    def _train_oversampled(self, data: np.ndarray, model: bool = True) \
            -> [np.ndarray, (np.ndarray, LinearDRModel)]:

        _data = self._check_data(data)
        _model = bool(model)

        mean = np.mean(data, axis=0)
        xc = data - mean

        npts, ndim = _data.shape
        assert npts >= (self.knn + 1)

        if self.ncomp is None:
            ncomp = min(xc.shape)
        elif self.ncomp >= 1:
            ncomp = self.ncomp
        else:
            raise ValueError

        aff_mtx = _compute_affinity_mtx(xc,
                                        knn=self.knn,
                                        weighted=self.weighted,
                                        sig=self.sig)

        l_mtx, d_vct = sp.csgraph.laplacian(aff_mtx, return_diag=True)
        xlx_mtx = np.dot(xc.T, l_mtx @ xc)
        xdx_mtx = np.dot(xc.T, d_vct[:, None] * xc)
        evals, evcts = la.eigh(xlx_mtx, b=xdx_mtx)

        inv_basis = evcts[:, :ncomp]

        z = np.dot(xc, inv_basis)

        if model:

            basis = la.pinv(inv_basis)
            error = la.norm(xc - z.dot(basis)) / xc.size**0.5

            info = {
                'model': 'LPP',
                'error': error,
                'knn': self.knn,
                'weighted': self.weighted,
                'sig': self.sig,
            }

            lpp_model = LinearDRModel(basis, inv_basis=inv_basis, info=info, xoffset=mean)

            return z, lpp_model

        else:
            return z

    def _train_undersampled(self, data: np.ndarray, model: bool = True) \
            -> [np.ndarray, (np.ndarray, LinearDRModel)]:

        _data = self._check_data(data)
        _model = bool(model)

        mean = np.mean(data, axis=0)
        xc = data - mean

        npts, ndim = _data.shape
        assert npts >= (self.knn + 1)

        # noinspection PyTupleAssignmentBalance
        u_mtx, s_vct, vt_mtx = la.svd(xc, full_matrices=False)

        if self.rcond is None:
            rcond = max(xc.shape) * np.finfo(s_vct.dtype).eps
        else:
            rcond = self.rcond
        rank = s_vct.size - np.searchsorted(s_vct[::-1], s_vct[0] * rcond)

        if self.ncomp is None:
            ncomp = rank
        elif self.ncomp >= 1:
            assert self.ncomp <= rank
            ncomp = self.ncomp
        else:
            raise ValueError

        total_var = np.sum(s_vct**2)

        u_mtx = u_mtx[:, :ncomp]
        s_vct = s_vct[:ncomp]
        vt_mtx = vt_mtx[:ncomp, :]

        aff_mtx = _compute_affinity_mtx(xc,
                                        knn=self.knn,
                                        weighted=self.weighted,
                                        sig=self.sig)

        l_mtx, d_vct = sp.csgraph.laplacian(aff_mtx, return_diag=True)

        ulu_mtx = np.dot(u_mtx.T, l_mtx @ u_mtx)
        udu_mtx = np.dot(u_mtx.T, d_vct[:, None] * u_mtx)
        evals, evcts = la.eigh(ulu_mtx, b=udu_mtx)

        z = np.dot(u_mtx, evcts)

        if model:

            inv_basis = np.dot(vt_mtx.T, evcts / s_vct[:, None])
            basis = np.dot(la.pinv(evcts) * s_vct[None, :], vt_mtx)

            error = np.sqrt((total_var - np.sum(s_vct**2)) / xc.size)

            info = {
                'model': 'LPP',
                'error': error,
                'knn': self.knn,
                'weighted': self.weighted,
                'sig': self.sig,
            }

            lpp_model = LinearDRModel(basis, inv_basis=inv_basis, info=info, xoffset=mean)

            return z, lpp_model

        else:
            return z
