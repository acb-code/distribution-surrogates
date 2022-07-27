import numpy as np
import os


direction_file = os.path.join(os.path.dirname(__file__), 'new-joe-kuo-6.21201')


def sobol_points(npts, ndim):

    lbits = np.ceil(np.log(npts)/np.log(2.)).astype(np.uint32)

    c_vtr = np.ones(npts, dtype=np.uint32)

    for i in range(1, npts):
        value = i
        while value & 1:
            value >>= 1
            c_vtr[i] += 1

    pts_mtx = np.zeros((npts, ndim), dtype=np.float64)

    v_vtr = np.zeros(lbits + 1, dtype=np.uint32)

    for i in range(1, lbits + 1):
        v_vtr[i] = 1 << (32 - i)

    x_vtr = np.zeros(npts, dtype=np.uint32)

    for i in range(1, npts):
        x_vtr[i] = x_vtr[i-1] ^ v_vtr[c_vtr[i-1]]
        pts_mtx[i, 0] = np.float64(x_vtr[i])/(2**32)

    with open(direction_file) as fid:
        fid.readline()

        for j in range(1, ndim):
            _, s, a, *m = [int(f) for f in fid.readline().split()]

            if lbits <= s:
                for i in range(1, lbits + 1):
                    v_vtr[i] = m[i-1] << (32 - i)

            else:
                for i in range(1, s + 1):
                    v_vtr[i] = m[i-1] << (32 - i)
                for i in range(s + 1, lbits + 1):
                    v_vtr[i] = v_vtr[i-s] ^ (v_vtr[i-s] >> s)
                    for k in range(1, s):
                        v_vtr[i] ^= ((a >> (s-1-k)) & 1) * v_vtr[i-k]

            for i in range(1, npts):
                x_vtr[i] = x_vtr[i-1] ^ v_vtr[c_vtr[i-1]]
                pts_mtx[i, j] = np.float64(x_vtr[i])/(2**32)

    return pts_mtx
