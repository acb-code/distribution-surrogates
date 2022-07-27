import numpy as np
import scipy.linalg as la

from scipy import sparse as sp
from scipy.spatial.distance import squareform, pdist

_EPS = np.finfo(np.float).eps


def _robust_solve(a, b, overwrite_a=False, overwrite_b=False, **kwargs):
    try:
        x = la.solve(a, b, overwrite_a=overwrite_a, overwrite_b=overwrite_b, **kwargs)
    except la.LinAlgError:
        x, *_ = la.lstsq(a, b, overwrite_a=overwrite_a, overwrite_b=overwrite_b, lapack_driver='gelsy')

    return x


def _robust_solve_triangular(a, b, overwrite_b=False, **kwargs):
    try:
        x = la.solve_triangular(a, b, overwrite_b=overwrite_b, **kwargs)
    except la.LinAlgError:
        x, *_ = la.lstsq(a, b, overwrite_b=overwrite_b, lapack_driver='gelsy')

    return x


def _svd_preproc(data):

    # noinspection PyTupleAssignmentBalance
    left_mtx, svals, right_mtx = la.svd(data, full_matrices=False)

    rank = svals.size - np.searchsorted(svals[::-1], svals[0] * _EPS)

    coeff = left_mtx[:, :rank].copy()
    svals = svals[:rank].copy()
    basis = right_mtx[:rank].copy()

    return coeff, svals, basis


def _orthonorm_basis(coeff, basis):

    # noinspection PyTupleAssignmentBalance
    zpca, zsvals, zrot = la.svd(coeff, full_matrices=False)

    wtmp = zsvals[:, None] * np.dot(zrot, basis)

    # noinspection PyTupleAssignmentBalance
    wrot, svals, wpca = la.svd(wtmp, full_matrices=False)

    zpca = np.dot(zpca, wrot)

    return zpca, svals, wpca


def _compute_affinity_mtx(x, knn=2, weighted=False, sig=1.):

    npts = x.shape[0]

    dist_mtx = squareform(pdist(x, metric='sqeuclidean'))
    dist_mtx[np.diag_indices_from(dist_mtx)] = np.inf

    jdx = np.argpartition(dist_mtx, knn, axis=1)[:, :knn].ravel()
    idx = np.repeat(np.arange(npts), knn)

    aff_mtx = sp.lil_matrix((npts, npts))

    if weighted:
        if sig is None:
            sig = dist_mtx[np.triu_indices(npts, 1)].var()
        weights = np.exp(-dist_mtx[idx, jdx] / sig)

        aff_mtx[idx, jdx] = weights
        aff_mtx[jdx, idx] = weights

    else:
        aff_mtx[idx, jdx] = 1.
        aff_mtx[jdx, idx] = 1.

    return aff_mtx.tocoo()
