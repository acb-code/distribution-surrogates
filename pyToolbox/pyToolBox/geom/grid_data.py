import copy as copy_module
import numpy as np

from .grid_functions import (
    cell_normals_line, node_normals_line,
    cell_normals_triangle, node_normals_triangle,
    cell_normals_quad, node_normals_quad,
    point_to_cell, cell_to_point
)


class GridData(object):

    _CELLTYPE_ENUM = {
        3: {'name': 'LINE', 'npts': 2},
        5: {'name': 'TRIANGLE', 'npts': 3},
        9: {'name': 'QUAD', 'npts': 4},
        10: {'name': 'TETRA', 'npts': 4},
        12: {'name': 'HEXAHEDRON', 'npts': 8},
        13: {'name': 'WEDGE', 'npts': 6},
        14: {'name': 'PYRAMID', 'npts': 5},
    }

    def __init__(self):

        self.points = None          # type: [None, np.ndarray]
        self.point_ids = None       # type: [None, np.ndarray]
        self.cell_type = None      # type: [None, np.ndarray, int]

        self.point_data = {}    # type: dict
        self.cell_data = {}     # type: dict

    @property
    def num_points(self) -> int:
        return 0 if self.points is None else self.points.shape[0]

    @property
    def num_cells(self) -> int:
        return 0 if self.point_ids is None else self.point_ids.shape[0]

    @property
    def is_2d(self) -> bool:
        return False if self.points is None else np.all(self.points[:, 2] == 0.)

    @property
    def is_polygon(self) -> bool:
        return {self.cell_type}.issubset([3, 5, 9])

    @property
    def num_point_arrays(self) -> int:
        return len(self.point_data)

    @property
    def num_cell_arrays(self) -> int:
        return len(self.cell_data)

    def set_points(self, points):
        _points = np.ascontiguousarray(points, dtype=np.float64)
        assert _points.ndim == 2 and _points.shape[1] == 3
        assert _points.dtype.kind == 'f'

        self.points = _points

    def set_cells(self, point_ids, cell_type):
        _point_ids = np.ascontiguousarray(point_ids, dtype=np.int64)
        _cell_type = int(cell_type)

        assert _cell_type in self._CELLTYPE_ENUM.keys()
        _cell_npts = self._CELLTYPE_ENUM[_cell_type]['npts']

        assert _point_ids.ndim == 2 and _point_ids.shape[1] == _cell_npts

        self.point_ids = _point_ids
        self.cell_type = _cell_type

    def add_point_array(self, name: str, array: np.ndarray, copy: bool = False) -> None:

        assert isinstance(name, str)

        _arr = np.array(array, copy=copy)
        assert _arr.dtype.kind == 'f'

        if _arr.ndim == 1:
            assert _arr.size == self.num_points
        elif _arr.ndim == 2:
            assert _arr.shape == (self.num_points, 3)
        else:
            raise ValueError

        self.point_data[name] = _arr

    def add_cell_array(self, name: str, array: np.ndarray, copy: bool = False) -> None:

        assert isinstance(name, str)

        _arr = np.array(array, copy=copy)
        assert _arr.dtype.kind == 'f'

        if _arr.ndim == 1:
            assert _arr.size == self.num_cells
        elif _arr.ndim == 2:
            assert _arr.shape == (self.num_cells, 3)
        else:
            raise ValueError

        self.cell_data[name] = _arr

    def clear(self) -> None:
        self.point_data = {}
        self.cell_data = {}

    def copy(self):
        return copy_module.deepcopy(self)

    def save_vtk(self, filename: str, overwrite: bool = True,
                 binary: bool = True, xml: bool = True) -> None:

        from pyToolBox.io import write_vtk_file
        write_vtk_file(filename, self, overwrite, binary, xml)

    def load_vtk(self, filename: str, exclude_fields: [None, list] = None, vtk_format: str = 'auto'):

        from pyToolBox.io import read_vtk_file
        grid_tmp = read_vtk_file(filename, exclude_fields, vtk_format)
        self.__dict__.update(grid_tmp.__dict__)

    def save_tecplot(self, filename: str,
                     overwrite: bool = True,
                     header: str = 'Tecplot output') -> None:

        from pyToolBox.io import write_tecplot_ascii
        write_tecplot_ascii(filename, self, overwrite, header)

    def compute_cell_center(self, store_value: bool = False) -> np.ndarray:

        assert self.points is not None
        assert self.is_polygon

        centroids = np.zeros((self.num_cells, 3), dtype=np.float64)
        point_to_cell(self.point_ids, self.points, centroids)

        if store_value:
            self.add_cell_array('CELL_CENTER', centroids)

        return centroids

    def compute_normals(self, store_value: bool = False, loc: str = 'node') -> np.ndarray:

        assert self.points is not None
        assert self.is_polygon
        assert loc in ('node', 'cell')

        is_node = loc == 'node'

        _points = self.points
        _pts_ids = self.point_ids

        if self.cell_type == 3 and self.is_2d:  # Lines (2D only)
            if is_node:
                normals = node_normals_line(_points, _pts_ids)
            else:
                normals = cell_normals_line(_points, _pts_ids)

        elif self.cell_type == 5:  # Triangles
            if is_node:
                normals = node_normals_triangle(_points, _pts_ids)
            else:
                normals = cell_normals_triangle(_points, _pts_ids)

        elif self.cell_type == 9:  # Quads
            if is_node:
                normals = node_normals_quad(_points, _pts_ids)
            else:
                normals = cell_normals_quad(_points, _pts_ids)

        else:
            raise NotImplementedError

        if is_node and store_value:
            self.add_point_array('NODE_NORMAL', normals)

        elif not is_node and store_value:
            self.add_cell_array('CELL_NORMAL', normals)

        return normals

    def compute_areas(self, store_value: bool = False, loc: str = 'node') -> np.ndarray:

        assert self.points is not None
        assert self.is_polygon

        is_node = loc == 'node'

        if is_node and 'NODE_NORMAL' in self.point_data:
            normals = self.point_data['NODE_NORMAL']

        elif not is_node and 'CELL_NORMAL' in self.cell_data:
            normals = self.cell_data['CELL_NORMAL']

        else:
            normals = self.compute_normals(loc=loc)

        areas = np.linalg.norm(normals, axis=-1)

        if is_node and store_value:
            self.add_point_array('NODE_AREA', areas)

        elif not is_node and store_value:
            self.add_cell_array('CELL_AREA', areas)

        return areas

    def compute_unit_normals(self, store_value: bool = False, loc: str = 'node') -> np.ndarray:

        assert self.points is not None
        assert self.is_polygon

        is_node = loc == 'node'

        if is_node and 'NODE_NORMAL' in self.point_data:
            normals = self.point_data['NODE_NORMAL'].copy()

        elif not is_node and 'CELL_NORMAL' in self.cell_data:
            normals = self.cell_data['CELL_NORMAL'].copy()

        else:
            normals = self.compute_normals(loc=loc)

        normals /= np.linalg.norm(normals, axis=-1)[:, None]

        if is_node and store_value:
            self.add_point_array('NODE_UNIT_NORMAL', normals)

        elif not is_node and store_value:
            self.add_cell_array('CELL_UNIT_NORMAL', normals)

        return normals

    def point_to_cell(self, *name: str, keep_old: bool = False):

        assert self.is_polygon

        if not name:
            name = [k for k in self.point_data.keys()]
        else:
            name = set(name).intersection(self.point_data.keys())

        for n in name:
            pts_arr = self.point_data[n]
            if pts_arr.ndim == 1:
                pts_arr = pts_arr[:, None]

            cell_arr = point_to_cell(self.point_ids, pts_arr)

            if cell_arr.shape[1] == 1:
                cell_arr = cell_arr.ravel()

            self.add_cell_array(n, cell_arr)

            if not keep_old:
                del self.point_data[n]

    def cell_to_point(self, *name: str, keep_old: bool = False):

        assert self.is_polygon

        if not name:
            name = [k for k in self.cell_data.keys()]
        else:
            name = set(name).intersection(self.cell_data.keys())

        for n in name:
            cell_arr = self.cell_data[n]
            if cell_arr.ndim == 1:
                cell_arr = cell_arr[:, None]

            pts_arr = np.zeros((self.num_points, cell_arr.shape[1]))
            pts_arr = cell_to_point(self.point_ids, cell_arr, pts_arr)

            if pts_arr.shape[1] == 1:
                pts_arr = pts_arr.ravel()

            self.add_point_array(n, pts_arr)

            if not keep_old:
                del self.cell_data[n]

    def integrate_data(self, *name: str, loc='node') -> dict:

        assert self.points is not None
        assert self.is_polygon

        is_node = loc == 'node'

        if is_node and 'NODE_AREA' in self.point_data:
            areas = self.point_data['NODE_AREA']

        elif not is_node and 'CELL_AREA' in self.cell_data:
            areas = self.cell_data['CELL_AREA']

        else:
            areas = self.compute_areas(loc=loc)

        results = {'AREA': np.sum(areas)}
        data = self.point_data if is_node else self.cell_data

        if not name:
            name = [k for k in data.keys()]
        else:
            name = set(name).intersection(data.keys())

        for n in name:
            array = data[n]
            results[n] = np.sum(areas * array.T, axis=-1)

        return results

    def triangulate(self):

        assert self.points is not None
        assert self.is_polygon

        if self.cell_type != 9:
            return

        old_pts_ids = self.point_ids

        new_n_cells = self.num_cells * 2
        new_pts_ids = np.zeros((new_n_cells, 3), dtype=np.int64)

        c0 = old_pts_ids[:, 0]
        c1 = old_pts_ids[:, 1]
        c2 = old_pts_ids[:, 2]
        c3 = old_pts_ids[:, 3]

        new_pts_ids[::2, 0] = c0
        new_pts_ids[::2, 1] = c1
        new_pts_ids[::2, 2] = c2
        new_pts_ids[1::2, 0] = c2
        new_pts_ids[1::2, 1] = c3
        new_pts_ids[1::2, 2] = c0

        self.point_ids = new_pts_ids
        self.cell_type = 5

        for k, arr in self.cell_data.items():

            if arr.ndim == 1:
                tmp = np.zeros(new_n_cells)
            else:
                tmp = np.zeros((new_n_cells, 3))

            tmp[::2] = arr/2
            tmp[1::2] = arr/2

            self.cell_data[k] = tmp
