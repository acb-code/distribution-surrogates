import numpy as np
import scipy.linalg as la

from .linear_model import LinearDRBuilder, LinearDRModel
from scipy.sparse.linalg import svds


class PCABuilder(LinearDRBuilder):

    def __init__(self,
                 ncomp: [None, int, float] = None,
                 method: str = 'svd',
                 maxiter: int = 1000):

        super().__init__(ncomp)

        assert method in ('svd', 'lanczos')

        assert maxiter > 0 and float(maxiter).is_integer()
        _maxiter = int(maxiter)

        self.method = method        # type: str
        self.maxiter = _maxiter     # type: int

    def train(self, data: np.ndarray, model: bool = True) \
            -> [np.ndarray, (np.ndarray, LinearDRModel)]:

        _data = self._check_data(data)
        _model = bool(model)

        if self.ncomp is not None:
            assert self.ncomp <= min(_data.shape)

        if self.method == 'lanczos':
            pca_out = self._train_lanczos(_data, _model)
        else:
            pca_out = self._train_svd(_data, _model)

        return pca_out

    def _train_svd(self, data: np.ndarray, model: bool = True) \
            -> [np.ndarray, (np.ndarray, LinearDRModel)]:
        pass

        npts, ndim = data.shape

        mean = np.mean(data, axis=0)
        xc = data - mean

        # noinspection PyTupleAssignmentBalance
        z, svals, basis = la.svd(xc, full_matrices=False)

        var_vct = (svals ** 2.) / npts
        var_total = var_vct.sum()

        if self.ncomp is None or self.ncomp < 1:
            train_ric = 1.0 if self.ncomp is None else self.ncomp

            ric_vct = np.cumsum(var_vct)
            ric_vct /= ric_vct[-1]
            ncomp = np.searchsorted(ric_vct, train_ric) + 1

        else:
            ncomp = self.ncomp

        z = z[:, :ncomp].copy() * svals[None, :ncomp]

        if model:
            var_vct = var_vct[:ncomp].copy()
            var_sum = var_vct.sum()
            train_ric = var_sum / var_total
            error = np.sqrt(max(var_total - var_sum, 0) / ndim)

            basis = basis[:ncomp].copy()

            info = {
                'model': 'PCA',
                'method': self.method,
                'RIC': train_ric,
                'var': var_vct,
                'error': error,
            }

            pca_model = LinearDRModel(basis, orthonorm=True, info=info, xoffset=mean)

            return z, pca_model

        else:
            return z

    def _train_lanczos(self, data: np.ndarray, model: bool = True) \
            -> [np.ndarray, (np.ndarray, LinearDRModel)]:

        npts, ndim = data.shape

        if self.ncomp is None:
            ncomp = min(npts, ndim) - 1
        elif self.ncomp < 1:
            raise ValueError
        else:
            assert self.ncomp < min(data.shape) - 1
            ncomp = self.ncomp

        mean = np.mean(data, axis=0)
        xc = data - mean
        var_total = la.norm(xc)**2 / npts

        z, svals, basis = svds(xc, k=ncomp, which='LM', maxiter=self.maxiter,
                               return_singular_vectors=True)

        idx_sort = np.argsort(svals)[::-1]
        svals[...] = svals[idx_sort]
        z[...] = z[:, idx_sort]

        z *= svals

        if model:

            basis[...] = basis[idx_sort, :]

            var_vct = (svals ** 2.) / npts
            var_sum = var_vct.sum()
            ric = var_sum / var_total
            error = np.sqrt((var_total - var_sum) / ndim)

            info = {
                'model': 'PCA',
                'method': self.method,
                'RIC': ric,
                'var': var_vct,
                'error': error
            }

            pca_model = LinearDRModel(basis, orthonorm=True, info=info, xoffset=mean)

            return z, pca_model

        else:
            return z
