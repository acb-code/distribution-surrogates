import unittest
import numpy as np

from pyToolBox.geom import FFDBox


class TestFFDTransform(unittest.TestCase):

    def setUp(self) -> None:

        self.corner_pts = np.array([[-2.1, -2., -0.5],
                                    [-2.1, -2.,  0.5],
                                    [-1.1, 1.5, -0.5],
                                    [-1.1, 1.5,  0.5],
                                    [ 1.1, -1., -0.5],
                                    [ 1.1, -1.,  0.5],
                                    [ 1.1,  1., -0.5],
                                    [ 1.1,  1.,  0.5]])

        self.geom_pts = np.array([[ 1.,  0.,  0.],
                                  [ 0.,  1.,  0.],
                                  [-1.,  0.,  0.],
                                  [-0., -1.,  0.],
                                  [ 1., -0.,  0.]])

        self.box = FFDBox(self.corner_pts, (5, 5, 0))

    def test_set_corners(self):

        box, corner_pts = self.box, self.corner_pts
        box.set_corners(corner_pts, mode='ijk')
        self.assertTrue(np.all(box.corner_pts == corner_pts))

        ijk2vtk = np.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=int)
        box.set_corners(corner_pts[ijk2vtk], mode='vtk')
        self.assertTrue(np.all(box.corner_pts == corner_pts))

        _corner_pts = corner_pts.copy()
        _corner_pts[3] = np.array([-1, -1, -0.5])
        self.assertRaises(ValueError, box.set_corners, _corner_pts)

    def test_set_degree(self):

        box = self.box
        box.set_degree(5, 5, 0)

        self.assertEqual(box.ctrl_pts.size, 108)

    def test_geom2param(self):

        box, geom_pts = self.box, self.geom_pts
        param_pts = box.geom2param(geom_pts)

        _geom_pts = box.param2geom(param_pts)
        self.assertTrue(np.allclose(_geom_pts, geom_pts))
        self.assertTrue(np.all(param_pts != geom_pts))

        big_geom_pts = 2 * geom_pts
        self.assertWarns(Warning, box.geom2param, big_geom_pts)


if __name__ == '__main__':
    unittest.main()
