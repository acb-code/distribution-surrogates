import numpy as np

from abc import ABC, abstractmethod


class BaseFFDVariable(ABC):

    @abstractmethod
    def __init__(self, i_ind: [None, int], j_ind: [None, int], k_ind: [None, int]):

        if i_ind is None:
            i_ind = slice(None)
        else:
            assert isinstance(i_ind, (int, np.integer)) and i_ind >= 0

        if j_ind is None:
            j_ind = slice(None)
        else:
            assert isinstance(j_ind, (int, np.integer)) and j_ind >= 0

        if k_ind is None:
            k_ind = slice(None)
        else:
            assert isinstance(k_ind, (int, np.integer)) and k_ind >= 0

        self.ffd_slice = np.s_[i_ind, j_ind, k_ind]

    @property
    def name(self):
        return self.__class__.__name__

    def apply(self, pts_orig: np.ndarray, value: float, pts_def: np.ndarray) -> None:
        pass


class BaseFFDVariable2D(BaseFFDVariable, ABC):

    def __init__(self, i_ind: [None, int], j_ind: [None, int]):
        super().__init__(i_ind, j_ind, 0)


class FFDAngleOfAttack(BaseFFDVariable):

    def __init__(self):
        super().__init__(None, None, None)


class FFDCtrlPoint(BaseFFDVariable):

    def __init__(self, i_ind: int, j_ind: int, k_ind: int,
                 x_mov: float, y_mov: float, z_mov: float):
        super().__init__(i_ind, j_ind, k_ind)

        assert i_ind is not None
        assert j_ind is not None
        assert k_ind is not None

        self.move = np.array([x_mov, y_mov, z_mov], dtype=float)

    def apply(self, pts_orig: np.ndarray, value: float, pts_def: np.ndarray) -> None:
        pts_def[self.ffd_slice] += value * self.move


class FFDCamber(BaseFFDVariable):

    def __init__(self, i_ind: int, j_ind: int):
        super().__init__(i_ind, j_ind, None)

        assert i_ind is not None
        assert j_ind is not None

    def apply(self, pts_orig: np.ndarray, value: float, pts_def: np.ndarray) -> None:
        slice_move = np.copy(pts_orig[self.ffd_slice][-1] - pts_orig[self.ffd_slice][0])
        slice_move /= np.linalg.norm(slice_move)
        pts_def[self.ffd_slice] += value * slice_move


class FFDThickness(BaseFFDVariable):

    def __init__(self, i_ind: int, j_ind: int):
        super().__init__(i_ind, j_ind, None)

        assert i_ind is not None
        assert j_ind is not None

    def apply(self, pts_orig: np.ndarray, value: float, pts_def: np.ndarray) -> None:
        slice_move = np.copy(pts_orig[self.ffd_slice])
        slice_move -= 0.5 * (slice_move[-1] + slice_move[0])
        slice_move /= np.linalg.norm(slice_move[-1] - slice_move[0])
        pts_def[self.ffd_slice] += 2 * value * slice_move


class FFDTwist(BaseFFDVariable):

    def __init__(self, j_ind: int,
                 x_start: float, y_start: float, z_start: float,
                 x_end: float, y_end: float, z_end: float,
                 ref_length: float = 1.):
        super().__init__(None, j_ind, None)

        assert j_ind is not None

        self.ref_length = float(ref_length)
        self.start = np.array([x_start, y_start, z_start], dtype=float)
        self.end = np.array([x_end, y_end, z_end], dtype=float)

    def apply(self, pts_orig: np.ndarray, value: float, pts_def: np.ndarray) -> None:

        # note: this definition of theta is consistent with SU2
        theta = np.arctan(value/self.ref_length)

        axis = self.end - self.start
        axis /= np.linalg.norm(axis)  # unit vector

        slice_pts = pts_orig[self.ffd_slice] - self.start

        slice_pts_rot = (
            (1. - np.cos(theta)) * np.dot(slice_pts, axis)[..., None] * axis
            + np.cos(theta) * slice_pts
            + np.sin(theta) * np.cross(axis, slice_pts)
        )

        pts_def[self.ffd_slice] += slice_pts_rot - slice_pts


class FFDCtrlPoint2D(FFDCtrlPoint, BaseFFDVariable2D):

    def __init__(self, i_ind: int, j_ind: int, x_mov: float, y_mov: float):
        BaseFFDVariable2D.__init__(self, i_ind, j_ind)
        self.move = np.array([x_mov, y_mov, 0], dtype=float)


class FFDCamber2D(FFDCamber, BaseFFDVariable2D):

    def __init__(self, i_ind: int):
        BaseFFDVariable2D.__init__(self, i_ind, None)
        assert i_ind is not None


class FFDThickness2D(FFDThickness, BaseFFDVariable2D):

    def __init__(self, i_ind: int):
        BaseFFDVariable2D.__init__(self, i_ind, None)
        assert i_ind is not None
