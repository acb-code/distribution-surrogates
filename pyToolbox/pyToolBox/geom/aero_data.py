import numpy as np
import copy as copy_module

from numpy import cos, sin
from collections import namedtuple
from ..geom import GridData


class AeroCoeff(namedtuple(
        'AeroCoeff', [
            'CL', 'CD', 'CSF',
            'CFx', 'CFy', 'CFz',
            'CMx', 'CMy', 'CMz'
        ])
):
    __slots__ = ()

    def __new__(cls,
                CL=0., CD=0., CSF=0.,
                CFx=0., CFy=0., CFz=0.,
                CMx=0., CMy=0., CMz=0.):
        # noinspection PyTypeChecker,PyArgumentList
        return super().__new__(cls, CL, CD, CSF, CFx, CFy, CFz, CMx, CMy, CMz)

    def __repr__(self):
        return '\n'.join(['{:3} = {:g}'.format(k, val)
                          for k, val in self._asdict().items()])


class AeroGridData(GridData):

    def __init__(self, grid: [GridData, None] = None, copy: bool = False):
        super().__init__()

        if isinstance(grid, GridData):
            if copy:
                self.__dict__ = copy_module.deepcopy(grid.__dict__)
            else:
                self.__dict__ = grid.__dict__.copy()

        elif grid is not None:
            raise ValueError

        self._ref_length = 1.           # type: float
        self._ref_semispan = 1.             # type: float
        self._ref_area = 1.             # type: float
        self._ref_origin = np.zeros(3)  # type: np.ndarray

        self._ref_vel = 1.  # type: float
        self._ref_rho = 1.  # type: float

        self._alpha = 0.    # type: float
        self._beta = 0      # type: float

    @property
    def angle_of_attack(self) -> float:
        return self._alpha

    @angle_of_attack.setter
    def angle_of_attack(self, angle: float):
        self._alpha = float(angle)

    @property
    def sideslip_angle(self) -> float:
        return self._beta

    @sideslip_angle.setter
    def sideslip_angle(self, angle: float):
        self._beta = float(angle)

    @property
    def ref_length(self) -> float:
        return self._ref_length

    @ref_length.setter
    def ref_length(self, length: float):
        assert length > 0.
        self._ref_length = float(length)

    @property
    def ref_semispan(self) -> float:
        return self._ref_length

    @ref_semispan.setter
    def ref_semispan(self, span: float):
        assert span > 0.
        self._ref_semispan = float(span)

    @property
    def ref_area(self) -> float:
        return self._ref_area

    @ref_area.setter
    def ref_area(self, area: float):
        assert area > 0.
        self._ref_area = float(area)

    @property
    def ref_origin(self) -> np.ndarray:
        return self._ref_origin.copy()

    @property
    def ref_velocity(self) -> float:
        return self._ref_vel

    @ref_velocity.setter
    def ref_velocity(self, vel):
        self._ref_vel = float(vel)

    @property
    def ref_density(self) -> float:
        return self._ref_rho

    @ref_density.setter
    def ref_density(self, rho):
        self._ref_rho = float(rho)

    @ref_origin.setter
    def ref_origin(self, origin: np.ndarray):
        _origin = np.array(origin, copy=True, dtype=float).ravel()
        assert _origin.size == 3
        self._ref_origin = _origin

    def compute_aero_coeff(self,
                           cp_name: str = 'Pressure_Coefficient',
                           cf_name: str = 'Skin_Friction_Coefficient',
                           dimensional: bool = False,
                           data_loc: str = 'node',
                           breakdown: bool = False) -> [dict, (dict, dict, dict)]:

        assert self.points is not None
        assert self.is_polygon
        assert data_loc in ('node', 'cell')

        if data_loc == 'node':
            cp_arr = self.point_data.get(cp_name, None)
            cf_arr = self.point_data.get(cf_name, None)
            dist_arr = self.points - self._ref_origin

            if 'NODE_NORMAL' in self.point_data:
                normals = self.point_data['NODE_NORMAL']
            else:
                normals = self.compute_normals(loc=data_loc)

        else:
            cp_arr = self.cell_data.get(cp_name, None)
            cf_arr = self.cell_data.get(cf_name, None)

            if 'CELL_CENTER' in self.cell_data:
                dist_arr = self.cell_data['CELL_CENTER']
            else:
                dist_arr = self.compute_cell_center()

            if 'CELL_NORMAL' in self.cell_data:
                normals = self.cell_data['CELL_NORMAL']
            else:
                normals = self.compute_normals(loc=data_loc)

        is_2d = self.is_2d
        is_dimensional = bool(dimensional)
        ref_q = 0.5 * self._ref_rho * self._ref_vel ** 2

        if is_2d:
            rot_mtx = _body_to_wind_matrix_2d(self._alpha)
        else:
            rot_mtx = _body_to_wind_matrix(self._alpha, self._beta)

        if cp_arr is not None:
            force_arr = -normals * cp_arr[:, None]
            moment_arr = np.cross(dist_arr, force_arr)

            force_sum = np.sum(force_arr, axis=0) / self._ref_area
            moment_sum = np.sum(moment_arr, axis=0) / self._ref_area / self._ref_length

            if is_dimensional:
                force_sum /= ref_q
                moment_sum /= ref_q

            if is_2d:
                cd, cl = np.dot(rot_mtx, force_sum[:2])
                cc = 0.
            else:
                cd, cc, cl = np.dot(rot_mtx, force_sum)

            pressure_coeff = AeroCoeff(
                CL=cl, CFx=force_sum[0], CMx=moment_sum[0],
                CD=cd, CFy=force_sum[1], CMy=moment_sum[1],
                CSF=cc, CFz=force_sum[2], CMz=moment_sum[2],
            )
        else:
            pressure_coeff = AeroCoeff()

        if cf_arr is not None:

            areas = np.linalg.norm(normals, axis=-1)
            force_arr = areas[:, None] * cf_arr
            moment_arr = np.cross(dist_arr, force_arr)

            force_sum = np.sum(force_arr, axis=0) / self._ref_area
            moment_sum = np.sum(moment_arr, axis=0) / self._ref_area / self._ref_length

            if is_dimensional:
                force_sum /= ref_q
                moment_sum /= ref_q

            if is_2d:
                cd, cl = np.dot(rot_mtx, force_sum[:2])
                cc = 0.
            else:
                cd, cc, cl = np.dot(rot_mtx, force_sum)

            friction_coeff = AeroCoeff(
                CL=cl, CFx=force_sum[0], CMx=moment_sum[0],
                CD=cd, CFy=force_sum[1], CMy=moment_sum[1],
                CSF=cc, CFz=force_sum[2], CMz=moment_sum[2],
            )
        else:
            friction_coeff = AeroCoeff()

        total_coeff = AeroCoeff(*np.add(pressure_coeff, friction_coeff))

        if breakdown:
            return total_coeff, pressure_coeff, friction_coeff

        else:
            return total_coeff


def _body_to_wind_matrix(alpha: float, beta: float, degree=True):

    _a = np.deg2rad(alpha) if degree else alpha
    _b = np.deg2rad(beta) if degree else beta

    rot_mtx = np.array(
        [
            [ cos(_a) * cos(_b), sin(_b),  sin(_a) * cos(_b)],
            [-cos(_a) * sin(_b), cos(_b), -sin(_a) * sin(_b)],
            [          -sin(_a),      0.,            cos(_a)],
        ]
    )

    return rot_mtx


def _body_to_wind_matrix_2d(alpha: float, degree=True):

    _a = np.deg2rad(alpha) if degree else alpha

    rot_mtx = np.array(
        [
            [ cos(_a), sin(_a)],
            [-sin(_a), cos(_a)],
        ]
    )

    return rot_mtx
