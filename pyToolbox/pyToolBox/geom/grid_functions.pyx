#!python
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True

import numpy as np

from cython cimport floating
from libc.stdint cimport int64_t


cpdef cell_normals_line(double[:, ::1] points, int64_t[:, ::1] point_ids, double[:, ::1] out=None):

    cdef Py_ssize_t n_cells, i
    cdef int64_t n0, n1
    cdef double nrm0, nrm1

    n_cells = point_ids.shape[0]

    if out is None:
        out = np.zeros((n_cells, 3), dtype=np.float64)

    for i in range(n_cells):
        n0, n1 = point_ids[i, 0], point_ids[i, 1]

        nrm0 = (points[n0, 1] - points[n1, 1]) / 2.
        nrm1 = (points[n1, 0] - points[n0, 0]) / 2.

        out[i, 0] = nrm0
        out[i, 1] = nrm1
        out[i, 2] = 0.

    return np.asarray(out)


cpdef node_normals_line(double[:, ::1] points, int64_t[:, ::1] point_ids, double[:, ::1] out=None):

    cdef Py_ssize_t n_cells, n_nodes, i
    cdef int64_t n0, n1
    cdef double nrm0, nrm1

    n_cells = point_ids.shape[0]
    n_nodes = points.shape[0]

    if out is None:
        out = np.zeros((n_nodes, 3), dtype=np.float64)

    for i in range(n_cells):
        n0, n1 = point_ids[i, 0], point_ids[i, 1]

        nrm0 = (points[n0, 1] - points[n1, 1]) / 2.
        nrm1 = (points[n1, 0] - points[n0, 0]) / 2.

        out[n0, 0] += nrm0
        out[n1, 0] += nrm0
        out[n0, 1] += nrm1
        out[n1, 1] += nrm1

    return np.asarray(out)


cpdef cell_normals_triangle(double[:, ::1] points, int64_t[:, ::1] point_ids, double[:, ::1] out=None):

    cdef Py_ssize_t n_cells, i, j, d
    cdef int64_t n0, n1, n2
    cdef double[3] a, b, nrm

    n_cells = point_ids.shape[0]

    if out is None:
        out = np.zeros((n_cells, 3), dtype=np.float64)

    for i in range(n_cells):
        n0, n1, n2 = point_ids[i, 0], point_ids[i, 1], point_ids[i, 2]

        for d in range(3):
            a[d] = points[n1, d] - points[n0, d]
            b[d] = points[n2, d] - points[n0, d]

        _cross_vector(a, b, nrm)

        for j in range(3):
            out[i, j] = nrm[j] / 2.

    return np.asarray(out)


cpdef node_normals_triangle(double[:, ::1] points, int64_t[:, ::1] point_ids, double[:, ::1] out=None):

    cdef Py_ssize_t n_cells, n_nodes, i, j, d
    cdef int64_t n0, n1, n2
    cdef double[3] a, b, nrm

    n_cells = point_ids.shape[0]
    n_nodes = points.shape[0]

    if out is None:
        out = np.zeros((n_nodes, 3), dtype=np.float64)

    for i in range(n_cells):
        n0, n1, n2 = point_ids[i, 0], point_ids[i, 1], point_ids[i, 2]

        for d in range(3):
            a[d] = points[n1, d] - points[n0, d]
            b[d] = points[n2, d] - points[n0, d]

        _cross_vector(a, b, nrm)

        for j in range(3):
            out[n0, j] += nrm[j] / 6.
            out[n1, j] += nrm[j] / 6.
            out[n2, j] += nrm[j] / 6.

    return np.asarray(out)


cpdef cell_normals_quad(double[:, ::1] points, int64_t[:, ::1] point_ids, double[:, ::1] out=None):

    cdef Py_ssize_t n_cells, i, j, d
    cdef int64_t n0, n1
    cdef double[3] a, b, nrm, center

    n_cells = point_ids.shape[0]

    if out is None:
        out = np.zeros((n_cells, 3), dtype=np.float64)

    for i in range(n_cells):
        center[0], center[1], center[2] = 0., 0., 0.

        for j in range(4):
            n0 = point_ids[i, j]
            for d in range(3):
                center[d] += points[n0, d] / 4.

        for j in range(4):
            n0, n1 = point_ids[i, j], point_ids[i, (j + 1) % 4]

            for d in range(3):
                a[d] = points[n1, d] - points[n0, d]
                b[d] = center[d] - points[n0, d]

            _cross_vector(a, b, nrm)

            for d in range(3):
                out[i, d] += nrm[d] / 2.

    return np.asarray(out)


cpdef node_normals_quad(double[:, ::1] points, int64_t[:, ::1] point_ids, double[:, ::1] out=None):

    cdef Py_ssize_t n_cells, n_nodes, i, j, d
    cdef int64_t n0, n1
    cdef double[3] a, b, nrm, center

    n_cells = point_ids.shape[0]
    n_nodes = points.shape[0]

    if out is None:
        out = np.zeros((n_nodes, 3), dtype=np.float64)

    for i in range(n_cells):
        center[0], center[1], center[2] = 0., 0., 0.

        for j in range(4):
            n0 = point_ids[i, j]
            for d in range(3):
                center[d] += points[n0, d] / 4.

        for j in range(4):
            n0, n1 = point_ids[i, j], point_ids[i, (j + 1) % 4]

            for d in range(3):
                a[d] = points[n1, d] - points[n0, d]
                b[d] = center[d] - points[n0, d]

            _cross_vector(a, b, nrm)

            for d in range(3):
                out[n0, d] += nrm[d] / 4.
                out[n1, d] += nrm[d] / 4.

    return np.asarray(out)


cpdef point_to_cell(int64_t[:, ::1] point_ids, floating[:, ::1] point_arr, floating[:, ::1] out=None):

    cdef Py_ssize_t n_cells, n_nodes, n_dim, i, j, d
    cdef int64_t n0

    n_cells = point_ids.shape[0]
    n_nodes = point_ids.shape[1]
    n_dim = point_arr.shape[1]

    if out is None:
        out = np.zeros((n_cells, n_dim), dtype=np.asarray(point_arr).dtype)

    for i in range(n_cells):
        for j in range(n_nodes):
            n0 = point_ids[i, j]
            for d in range(n_dim):
                out[i, d] += point_arr[n0, d] / <floating>n_nodes

    return np.asarray(out)


cpdef cell_to_point(int64_t[:, ::1] point_ids, floating[:, ::1] cell_arr, floating[:, ::1] out=None):

    cdef Py_ssize_t n_cells, n_nodes, n_dim, i, j, d
    cdef int64_t n0
    cdef short[::1] pts_count

    n_cells = point_ids.shape[0]
    n_nodes = point_ids.shape[1]
    n_pts = out.shape[0]
    n_dim = out.shape[1]

    pts_count = np.zeros(n_pts, dtype=np.short)

    if out is None:
        out = np.zeros((n_pts, n_dim), dtype=np.asarray(cell_arr).dtype)

    for i in range(n_cells):
        for j in range(n_nodes):
            n0 = point_ids[i, j]
            for d in range(n_dim):
                out[n0, d] += cell_arr[i, d]
                pts_count[n0] += 1

    for i in range(n_pts):
        for d in range(n_dim):
            out[i, d] /= <floating>pts_count[i]

    return np.asarray(out)


cdef inline void _cross_vector(double[3] a, double[3] b, double[3] c) nogil:
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]