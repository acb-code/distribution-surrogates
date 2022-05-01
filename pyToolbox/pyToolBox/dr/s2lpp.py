import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .misc import _svd_preproc, _compute_affinity_mtx
from .linear_model import LinearS2DRBuilder, LinearDRModel


class S2LPPBuilder(LinearS2DRBuilder):

    def __init__(self,
                 ncomp: [None, int] = None,
                 knn: int = 2,
                 weighted: bool = False,
                 sig: [None, float] = None,
                 pca_preproc: bool = True):

        super().__init__(ncomp)

        assert float(knn).is_integer() and knn >= 1

        if sig is None:
            _sig = None
        else:
            assert sig > 0.
            _sig = float(sig)

        self.knn = int(knn)
        self.weighted = bool(weighted)
        self.sig = _sig
        self.pca_preproc = bool(pca_preproc)

    def train(self, data1: np.ndarray, data2: np.ndarray, model: bool = True)\
        -> [(np.ndarray, np.ndarray), (np.ndarray, np.ndarray, LinearDRModel, LinearDRModel)]:

        _data1 = self._check_data(data1)
        _data2 = self._check_data(data2)
        _model = bool(model)

        xmean = _data1.mean(axis=0)
        ymean = _data2.mean(axis=0)

        x = _data1 #- xmean
        y = _data2 #- ymean

        # x, pca_xcoeff, pca_xbasis = _svd_preproc(x)
        # y, pca_ycoeff, pca_ybasis = _svd_preproc(y)

        xnpts, xdim = x.shape
        ynpts, ydim = y.shape

        xaff_mtx = _compute_affinity_mtx(x,
                                         knn=self.knn,
                                         weighted=self.weighted,
                                         sig=self.sig)

        yaff_mtx = _compute_affinity_mtx(y,
                                         knn=self.knn,
                                         weighted=self.weighted,
                                         sig=self.sig)

        xyaff_mtx = sp.dia_matrix((1, [0]), shape=(xnpts, ynpts))

        xy = np.block([[x, np.zeros((xnpts, ydim))],
                       [np.zeros((ynpts, xdim)), y]])

        aff_mtx = sp.bmat([[xaff_mtx, xyaff_mtx],
                           [xyaff_mtx.T, yaff_mtx]])

        l_mtx, d_vct = sp.csgraph.laplacian(aff_mtx, return_diag=True)
        xylyx_mtx = np.dot(xy.T, l_mtx @ xy)
        xydyx_mtx = np.dot(xy.T, d_vct[:, None] * xy)
        evals, evcts = la.eigh(xylyx_mtx, b=xydyx_mtx)

        evals2, z2 = la.eigh(l_mtx.toarray(), b=np.diag(d_vct))

        inv_xbasis = evcts[:xdim]
        inv_ybasis = evcts[xdim:]

        z = np.dot(xy, evcts)

        return z