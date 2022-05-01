import unittest
import os
import numpy as np

from pyToolBox.geom import *

_file_naca0012_orig = os.path.join(os.path.dirname(__file__), 'NACA0012', 'NACA0012_orig.txt')
_file_naca0012_all = os.path.join(os.path.dirname(__file__), 'NACA0012', 'NACA0012_all.txt')
_file_naca0012_camber = os.path.join(os.path.dirname(__file__), 'NACA0012', 'NACA0012_camber.txt')
_file_naca0012_ctrlpt = os.path.join(os.path.dirname(__file__), 'NACA0012', 'NACA0012_ctrlpt.txt')
_file_naca0012_thickness = os.path.join(os.path.dirname(__file__), 'NACA0012', 'NACA0012_thickness.txt')

_file_oneram6_orig = os.path.join(os.path.dirname(__file__), 'ONERA_M6', 'ONERAM6_orig.txt')
_file_oneram6_all = os.path.join(os.path.dirname(__file__), 'ONERA_M6', 'ONERAM6_all.txt')
_file_oneram6_camber = os.path.join(os.path.dirname(__file__), 'ONERA_M6', 'ONERAM6_camber.txt')
_file_oneram6_ctrlpt = os.path.join(os.path.dirname(__file__), 'ONERA_M6', 'ONERAM6_ctrlpt.txt')
_file_oneram6_thickness = os.path.join(os.path.dirname(__file__), 'ONERA_M6', 'ONERAM6_thickness.txt')
_file_oneram6_twist = os.path.join(os.path.dirname(__file__), 'ONERA_M6', 'ONERAM6_twist.txt')


class TestNACA0012(unittest.TestCase):

    file_pts_orig = _file_naca0012_orig
    file_pts_all = _file_naca0012_all
    file_pts_camber = _file_naca0012_camber
    file_pts_ctrlpt = _file_naca0012_ctrlpt
    file_pts_thickness = _file_naca0012_thickness

    def setUp(self) -> None:

        corner_pts = np.array([[-0.0001, -0.07, 0.0],
                               [ 1.0001, -0.07, 0.0],
                               [ 1.0001,  0.07, 0.0],
                               [-0.0001,  0.07, 0.0]])

        box = FFDBox(corners=corner_pts, degree=(4, 1, 0), mode='vtk')
        box.add_variables(
            FFDCtrlPoint2D(1, 1, 0.1, 1.),
            FFDCamber2D(2),
            FFDThickness2D(3),
        )

        self.box = box
        self.pts_orig = np.loadtxt(self.file_pts_orig, delimiter=',')
        self.pts_param = box.geom2param(self.pts_orig)

    def test_no_deformation(self):

        pts_def = self.box.deform(self.pts_param, [0., 0., 0.])
        self.assertTrue(np.allclose(pts_def, self.pts_orig))

    def test_ctrl_pt(self):
        pts_ctrlpt = np.loadtxt(self.file_pts_ctrlpt, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0.1, 0., 0.])
        self.assertTrue(np.allclose(pts_def, pts_ctrlpt))

    def test_camber(self):
        pts_camber = np.loadtxt(self.file_pts_camber, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0., 0.1, 0.])
        self.assertTrue(np.allclose(pts_def, pts_camber))

    def test_thickness(self):
        pts_thickness = np.loadtxt(self.file_pts_thickness, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0., 0., 0.1])
        self.assertTrue(np.allclose(pts_def, pts_thickness))

    def test_all(self):
        pts_all = np.loadtxt(self.file_pts_all, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0.1, 0.1, 0.1])
        self.assertTrue(np.allclose(pts_def, pts_all))


class TestONERAM6(unittest.TestCase):

    file_pts_orig = _file_oneram6_orig
    file_pts_all = _file_oneram6_all
    file_pts_camber = _file_oneram6_camber
    file_pts_ctrlpt = _file_oneram6_ctrlpt
    file_pts_thickness = _file_oneram6_thickness
    file_pts_twist = _file_oneram6_twist

    def setUp(self) -> None:

        corner_pts = np.array([[-0.0403,     0., -0.04836],
                               [ 0.8463,     0., -0.04836],
                               [  1.209, 1.2896, -0.04836],
                               [ 0.6851, 1.2896, -0.04836],
                               [-0.0403,     0.,  0.04836],
                               [ 0.8463,     0.,  0.04836],
                               [  1.209, 1.2896,  0.04836],
                               [ 0.6851, 1.2896,  0.04836]])

        box = FFDBox(corners=corner_pts, degree=(2, 4, 1), mode='vtk')
        box.add_variables(
            FFDCtrlPoint(1, 1, 1, 0.1, 0.1, 1.),
            FFDCamber(1, 2),
            FFDThickness(1, 3),
            FFDTwist(4, 0.955374, 1., 0., 0.955374, 2., 0., ref_length=1.)
        )

        self.box = box
        self.pts_orig = np.loadtxt(self.file_pts_orig, delimiter=',')
        self.pts_param = box.geom2param(self.pts_orig)

    def test_no_deformation(self):

        pts_def = self.box.deform(self.pts_param, [0., 0., 0., 0.])
        self.assertTrue(np.allclose(pts_def, self.pts_orig, atol=1e-6, rtol=1e-8))

    def test_ctrl_pt(self):
        pts_ctrlpt = np.loadtxt(self.file_pts_ctrlpt, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0.1, 0., 0., 0.])
        self.assertTrue(np.allclose(pts_def, pts_ctrlpt, atol=1e-6, rtol=1e-8))

    def test_camber(self):
        pts_camber = np.loadtxt(self.file_pts_camber, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0., 0.1, 0., 0.])
        self.assertTrue(np.allclose(pts_def, pts_camber, atol=1e-6, rtol=1e-8))

    def test_thickness(self):
        pts_thickness = np.loadtxt(self.file_pts_thickness, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0., 0., 0.1, 0.])
        self.assertTrue(np.allclose(pts_def, pts_thickness, atol=1e-6, rtol=1e-8))

    def test_twist(self):
        pts_twist = np.loadtxt(self.file_pts_twist, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0., 0., 0., 0.2])
        self.assertTrue(np.allclose(pts_def, pts_twist, atol=1e-6, rtol=1e-8))

    def test_all(self):
        pts_all = np.loadtxt(self.file_pts_all, delimiter=',')
        pts_def = self.box.deform(self.pts_param, [0.1, 0.1, 0.1, 0.2])
        self.assertTrue(np.allclose(pts_def, pts_all, atol=1e-6, rtol=1e-8))


if __name__ == '__main__':
    unittest.main()