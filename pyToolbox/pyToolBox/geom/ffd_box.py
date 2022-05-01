import warnings
import numpy as np

from scipy.special import binom
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import root

from .ffd_var import BaseFFDVariable, BaseFFDVariable2D


class FFDBox(object):

    def __init__(self, corners: [np.ndarray, None] = None,
                 degree: [tuple, None] = None, mode: str = 'ijk'):

        pts_tmp = np.array([[0., 0., 0.],
                            [0., 0., 1.],
                            [0., 1., 0.],
                            [0., 1., 1.],
                            [1., 0., 0.],
                            [1., 0., 1.],
                            [1., 1., 0.],
                            [1., 1., 1.]])

        self.corner_pts = pts_tmp               # type: np.ndarray
        self.corner_pts_p = pts_tmp.copy()      # type: np.ndarray
        self.ctrl_pts = pts_tmp.copy()          # type: np.ndarray
        self.ctrl_pts_p = pts_tmp.copy()        # type: np.ndarray
        self.ctrl_pts_orig = pts_tmp.copy()     # type: np.ndarray

        self.ffd_degree = np.ones(3, dtype=int)     # type: np.ndarray
        self.ffd_variables = []                     # type: list

        if corners is not None:
            self.set_corners(corners, mode=mode)

        if degree is not None:
            _degree = np.asarray(degree)
            assert _degree.size == 3
            dx, dy, dz = _degree.ravel()
            self.set_degree(dx, dy, dz)

    @property
    def num_variables(self) -> int:
        return len(self.ffd_variables)

    @property
    def num_ctrl_pts(self) -> int:
        return self.ctrl_pts.shape[0]

    @property
    def is_2d(self) -> bool:

        flag = self.ffd_degree[2] == 0
        flag &= np.all(self.corner_pts[::2, :2] == self.corner_pts[1::2, :2])
        flag &= np.all((self.corner_pts[::2, 2] + self.corner_pts[1::2, 2]) == 0.)

        return flag

    def set_corners(self, corner_pts: np.ndarray, mode: str = 'ijk') -> None:

        _corner_pts = np.array(corner_pts, dtype=float, copy=True)
        assert _corner_pts.shape == (8, 3) or _corner_pts.shape == (4, 3)
        assert mode in ('ijk', 'vtk')

        if _corner_pts.shape == (4, 3):
            assert np.all(_corner_pts[4:] == 0.)
            _corner_pts = np.vstack((_corner_pts, np.zeros_like(_corner_pts)))

        if np.all(_corner_pts[:, 2] == 0.) and np.all(_corner_pts[4:] == 0.):
            _corner_pts[4:] = _corner_pts[:4]

            if mode == 'vtk':
                _corner_pts[:4, 2] = -0.5
                _corner_pts[4:, 2] = 0.5
            else:
                _corner_pts[:] = _corner_pts[[0, 4, 1, 5, 2, 6, 3, 7]]
                _corner_pts[::2, 2] = -0.5
                _corner_pts[1::2, 2] = 0.5

        if mode == 'vtk':
            _corner_pts[:] = _corner_pts[[0, 4, 3, 7, 1, 5, 2, 6]]

        # Check if FFD box is convex
        hull = ConvexHull(_corner_pts)
        if hull.vertices.size != 8:
            raise ValueError('Provided corner points do not form a convex hull')

        ctrl_pts = _bezier_volume(self.ctrl_pts_p, _corner_pts, (1, 1, 1))

        self.corner_pts = _corner_pts
        self.ctrl_pts[...] = ctrl_pts
        self.ctrl_pts_orig[...] = ctrl_pts

        self.ffd_variables = []

    def set_degree(self, di: int, dj: int, dk: int) -> None:

        assert isinstance(di, (int, np.integer)) and di > 0
        assert isinstance(dj, (int, np.integer)) and dj > 0
        assert isinstance(dk, (int, np.integer)) and dk >= 0

        ffd_degree = np.array([di, dj, dk], dtype=int)

        ctrl_pts_p = np.array([k for k in np.ndindex(*(ffd_degree + 1))], dtype=float)
        if dk == 0:
            ctrl_pts_p[:, :2] /= ctrl_pts_p[-1, :2]
            ctrl_pts_p[:, 2] = 0.5
        else:
            ctrl_pts_p /= ctrl_pts_p[-1]

        ctrl_pts = _bezier_volume(ctrl_pts_p, self.corner_pts, (1, 1, 1))

        self.ffd_degree = ffd_degree
        self.ctrl_pts = ctrl_pts
        self.ctrl_pts_p = ctrl_pts_p
        self.ctrl_pts_orig = ctrl_pts.copy()

        self.ffd_variables = []

    def add_variables(self, *var: BaseFFDVariable) -> None:

        is_2d = self.is_2d
        for v in var:
            assert isinstance(v, BaseFFDVariable)
            if isinstance(v, BaseFFDVariable2D) and not is_2d:
                raise ValueError
            self.ffd_variables.append(v)

    def clear(self):
        self.ffd_variables = []
        self.ctrl_pts[...] = self.ctrl_pts_orig

    def set_ffd(self, *value: float) -> None:

        value = value[0] if len(value) == 1 else value

        _value = np.asarray(value, dtype=float).ravel()
        assert _value.size == len(self.ffd_variables)

        si, sj, sk = self.ffd_degree + 1

        pts_orig = self.ctrl_pts_orig.reshape((si, sj, sk, 3))
        pts_orig.setflags(write=False)

        pts_def = self.ctrl_pts.reshape((si, sj, sk, 3))
        pts_def[...] = pts_orig

        for ffd_var, val in zip(self.ffd_variables, _value):
            ffd_var.apply(pts_orig, val, pts_def)

    def deform(self, param_pts: np.ndarray, value: [list, tuple, np.ndarray]) -> np.ndarray:

        self.set_ffd(*value)
        geom_pts = self.param2geom(param_pts)

        return geom_pts

    def geom2param(self, geom_pts: np.ndarray) -> np.ndarray:

        _geom_pts = np.asarray(geom_pts)
        assert _geom_pts.ndim == 2 and _geom_pts.shape[1] == 3

        # Check if FFD box fully enclose the provided points
        delaunay = Delaunay(self.corner_pts)
        if np.any(delaunay.find_simplex(_geom_pts) < 0):
            warnings.warn('Provided points are not fully enclosed by the FFD box', Warning)

        offset = self.corner_pts[0]
        linear_transf = np.dot(np.linalg.pinv(self.corner_pts - offset), self.corner_pts_p)

        param_pts = _geom_pts - offset
        param_pts = np.dot(param_pts, linear_transf, out=param_pts)

        sol = root(self._ffd_residuals, param_pts.ravel(),
                   args=(_geom_pts, ),
                   method='krylov',
                   options={
                       'xatol': 1e-10,
                       'maxiter': 500,
                       'disp': False,
                   })

        if not sol.success:
            warnings.warn('Not converged', Warning)

        param_pts = sol.x.reshape(_geom_pts.shape)

        return param_pts

    def param2geom(self, param_pts: np.ndarray) -> np.ndarray:

        _param_pts = np.asarray(param_pts)
        assert _param_pts.ndim == 2 and _param_pts.shape[1] == 3

        return _bezier_volume(_param_pts, self.ctrl_pts, self.ffd_degree)

    def _ffd_residuals(self, param_pts, geom_pts):

        _param_pts = param_pts.reshape(geom_pts.shape)
        _geom_pts = _bezier_volume(_param_pts, self.corner_pts, (1, 1, 1))
        residuals = _geom_pts - geom_pts

        return residuals.ravel()


def _bezier_volume(pts: np.ndarray, ctrl_pts: np.ndarray, degree: [tuple, list, np.ndarray]) -> np.ndarray:

    di, dj, dk = degree
    _ctrl_pts = ctrl_pts.reshape((di + 1, dj + 1, dk + 1, 3))

    geom = np.zeros_like(pts)

    for i, j, k in np.ndindex(di + 1, dj + 1, dk + 1):
        tmp = _bernstein_poly(pts[:, 0], i, di)
        tmp *= _bernstein_poly(pts[:, 1], j, dj)
        tmp *= _bernstein_poly(pts[:, 2], k, dk)

        geom += tmp[:, None] * _ctrl_pts[i, j, k]

    return geom


def _bernstein_poly(x: [float, np.ndarray], k: int, n: int) -> [float, np.ndarray]:
    return binom(n, k) * x**k * (1. - x)**(n - k)


def _d_bernstein_poly(x: [float, np.ndarray], k: int, n: int) -> [float, np.ndarray]:
    tmp = _bernstein_poly(x, k - 1, n - 1) if k > 0 else 0.
    tmp -= _bernstein_poly(x, k, n - 1) if k < n else 0.
    return n * tmp
