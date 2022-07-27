import unittest
import os
import tempfile
import numpy as np
from pyToolBox.io import read_vtk_file, write_vtk_file

quad_file = os.path.join(os.path.dirname(__file__), 'unstruct_quad.vtk')
tri_file = os.path.join(os.path.dirname(__file__), 'unstruct_tri.vtk')
mixed_file = os.path.join(os.path.dirname(__file__), 'unstruct_mixed.vtk')

tmp_dir = tempfile.mkdtemp()
tmp_vtk = os.path.join(tmp_dir, 'tmp_file.vtk')


class TestReadVTK(unittest.TestCase):

    def test_quad(self):

        grid_data = read_vtk_file(quad_file)

        self.assertEqual(grid_data.cell_type, 9)
        self.assertEqual(grid_data.num_points, 8)
        self.assertEqual(grid_data.num_cells, 3)
        self.assertEqual(grid_data.num_point_arrays, 1)
        self.assertEqual(grid_data.num_cell_arrays, 1)

    def test_tri(self):

        grid_data = read_vtk_file(tri_file)

        self.assertEqual(grid_data.cell_type, 5)
        self.assertEqual(grid_data.num_points, 8)
        self.assertEqual(grid_data.num_cells, 6)
        self.assertEqual(grid_data.num_point_arrays, 1)
        self.assertEqual(grid_data.num_cell_arrays, 1)


class TestWriteVTK(unittest.TestCase):

    tmp_dir = tempfile.mkdtemp()
    tmp_vtk = os.path.join(tmp_dir, 'test_write_vtk.vtu')

    def test_quad(self):

        grid = read_vtk_file(quad_file)
        write_vtk_file(self.tmp_vtk, grid)
        grid_tmp = read_vtk_file(self.tmp_vtk)

        self.assertTrue(np.all(grid.points == grid_tmp.points))
        self.assertTrue(np.all(grid.point_ids == grid_tmp.point_ids))
        self.assertTrue(grid.cell_type == grid_tmp.cell_type)

        for p in grid.point_data:
            self.assertTrue(np.all(grid.point_data[p] == grid_tmp.point_data[p]))

        for c in grid.cell_data:
            self.assertTrue(np.all(grid.cell_data[c] == grid_tmp.cell_data[c]))

        os.remove(self.tmp_vtk)

    def test_tri(self):

        grid = read_vtk_file(tri_file)
        write_vtk_file(self.tmp_vtk, grid)
        grid_tmp = read_vtk_file(self.tmp_vtk)

        self.assertTrue(np.all(grid.points == grid_tmp.points))
        self.assertTrue(np.all(grid.point_ids == grid_tmp.point_ids))
        self.assertTrue(grid.cell_type == grid_tmp.cell_type)

        for p in grid.point_data:
            self.assertTrue(np.all(grid.point_data[p] == grid_tmp.point_data[p]))

        for c in grid.cell_data:
            self.assertTrue(np.all(grid.cell_data[c] == grid_tmp.cell_data[c]))

        os.remove(self.tmp_vtk)


if __name__ == '__main__':
    unittest.main()
