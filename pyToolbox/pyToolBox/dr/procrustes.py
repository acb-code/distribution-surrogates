import numpy as np
import scipy.linalg as la

from .linear_model import LinearS2DRBuilder, LinearDRModel, LinearDRBuilder
from . import PCABuilder, LPPBuilder


class ProcrustesBuilder(LinearS2DRBuilder):

    _available_dr = ['pca', 'lpp']

    def __init__(self,
                 ncomp: [None, float, int] = None,
                 ncomp_lo: [None, float, int] = None,
                 dr: str = 'pca',
                 **kwargs):

        super().__init__(ncomp=ncomp)

        if ncomp_lo is None:
            self.ncomp_lo = self.ncomp

        elif isinstance(ncomp_lo, (int, np.integer)):
            assert ncomp_lo > 0
            self.ncomp_lo = int(ncomp_lo)

        elif isinstance(ncomp_lo, (float, np.floating)):
            assert 0 < ncomp_lo <= 1
            self.ncomp_lo = float(ncomp_lo)

        else:
            raise ValueError

        assert str(dr).lower() in self._available_dr

        if dr == 'pca':
            self.dr_builder_hi = PCABuilder(ncomp=self.ncomp, **kwargs)     # type: LinearDRBuilder
            self.dr_builder_lo = PCABuilder(ncomp=self.ncomp_lo, **kwargs)  # type: LinearDRBuilder
        elif dr == 'lpp':
            self.dr_builder_hi = LPPBuilder(ncomp=self.ncomp, **kwargs)
            self.dr_builder_lo = LPPBuilder(ncomp=self.ncomp_lo, **kwargs)
        else:
            raise ValueError

    def train(self, data1: np.ndarray, data2: np.ndarray, model: bool = True)\
            -> [(np.ndarray, np.ndarray), (np.ndarray, np.ndarray, LinearDRModel, LinearDRModel)]:

        _data1 = self._check_data(data1)
        _data2 = self._check_data(data2)
        _model = bool(model)

        npts1, ndim1 = _data1.shape
        npts2, ndim2 = _data2.shape

        assert npts1 <= npts2

        if model:
            z1, dr1 = self.dr_builder_hi.train(_data1, model=model)
            z2, dr2 = self.dr_builder_lo.train(_data2, model=model)
        else:
            z1 = self.dr_builder_hi.train(_data1, model=model)
            z2 = self.dr_builder_lo.train(_data2, model=model)
            dr1 = dr2 = None

        z2l = z2[:npts1, :]

        z1_mean = z1.mean(axis=0)
        z2l_mean = z2l.mean(axis=0)

        _z1 = z1 - z1_mean
        z2 -= z2l_mean

        # noinspection PyTupleAssignmentBalance
        u, s, vt = la.svd(np.dot(z2l.T, _z1), full_matrices=False)

        p_mtx = np.dot(u, vt)

        scaling = np.sum(s) / la.norm(z2l)**2
        z2 = scaling * np.dot(z2, p_mtx)

        if model:

            error = la.norm(_z1 - z2[:npts1]) / npts1**0.5
            model_name = dr1.info.get('model', '')
            dr1.info.update(model=model_name + '+Procrustes')
            dr1.info.update(alignment_error=error)

            is_orthonormal2 = dr2.is_orthonormal
            basis2 = np.dot(p_mtx.T, dr2.basis)

            if is_orthonormal2:
                inv_basis2 = basis2.T
            elif dr2.inv_basis is not None:
                inv_basis2 = np.dot(dr2.inv_basis, p_mtx)
            else:
                inv_basis2 = None

            zoffset2 = z1_mean - scaling*np.dot(z2l_mean, p_mtx)

            info2 = dr2.info
            info2.update(model=model_name + '+Procrustes')
            info2.update(alignment_error=error)

            dr2 = LinearDRModel(basis2,
                                orthonorm=is_orthonormal2,
                                inv_basis=inv_basis2,
                                info=info2,
                                xoffset=dr2.xoffset, xscale=dr2.xscale,
                                zoffset=zoffset2, zscale=scaling)

            return z1, z2, dr1, dr2

        else:
            return z1, z2
