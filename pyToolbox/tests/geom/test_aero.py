import unittest
import os
import numpy as np

from pyToolBox.geom import AeroGridData
from pyToolBox.geom.aero_data import AeroCoeff


class TestRAE2822RANS(unittest.TestCase):

    filename = os.path.join(os.path.dirname(__file__), 'rae2822_rans.vtk')

    rtol = 1e-4

    angle_of_attack = 2
    ref_length = 1.
    ref_area = 1.
    ref_semispan = 1.
    ref_origin = [0.25, 0., 0.]

    integral_pts_res = {
        'Heat_Flux': -512.871850409766,
        'Pressure_Coefficient': -0.6052692197883294,
        'Skin_Friction_Coefficient': [
            0.00627512,
            0.00021781,
            0.
        ],
    }

    integral_cell_res = {
        'AREA': 2.027542340112933
    }

    aero_coeff = AeroCoeff(
        CL=0.6365145883,
        CD=0.01038198748,
        CSF=0.,
        CFx=-0.01183837572,
        CFy=0.6364891669,
        CFz=0.,
        CMx=0.,
        CMy=0.,
        CMz=0.08347798916,
    )

    def setUp(self) -> None:

        grid = AeroGridData()
        grid.load_vtk(self.filename)

        grid.angle_of_attack = self.angle_of_attack
        grid.ref_length = self.ref_length
        grid.ref_area = self.ref_area
        grid.ref_semispan = self.ref_semispan
        grid.ref_origin = self.ref_origin

        grid.compute_areas(store_value=True)
        grid.compute_normals(store_value=True, loc='cell')
        grid.compute_normals(store_value=True, loc='node')

        self.grid = grid

    def test_integral(self):

        pts_res = self.grid.integrate_data(loc='node')

        for k, val in self.integral_pts_res.items():
            self.assertTrue(
                np.allclose(val, pts_res[k]),
                'Computed integral of field "{}" is {}, expecting {}'
                .format(k, pts_res[k], val)
            )

    def test_aero_coeff(self):

        aero_coeff = self.grid.compute_aero_coeff()

        for i, c in enumerate(self.aero_coeff):
            self.assertTrue(
                np.allclose(c, aero_coeff[i], rtol=self.rtol),
                'Computed {} is {}, expecting {}'
                .format(aero_coeff._fields[i], aero_coeff[i], c)
            )

    def test_point_to_cell(self):

        grid = self.grid.copy()
        grid.point_to_cell('Pressure_Coefficient', 'Skin_Friction_Coefficient')

        aero_coeff = self.grid.compute_aero_coeff()

        for i, c in enumerate(self.aero_coeff):
            self.assertTrue(
                np.allclose(c, aero_coeff[i], rtol=self.rtol),
                'Computed {} is {}, expecting {}'
                .format(aero_coeff._fields[i], aero_coeff[i], c)
            )


class TestONERARANS(TestRAE2822RANS):

    filename = os.path.join(os.path.dirname(__file__), 'oneram6_rans_hexa.vtk')

    rtol = 1e-4

    angle_of_attack = 3.06
    ref_length = 0.64607
    ref_area = 0.7532
    ref_span = 1.1963
    ref_origin = [0.25, 0., 0.]

    integral_pts_res = {
        'Heat_Flux': -6785.985061259711,
        'Pressure_Coefficient': -0.3821443209900172,
        'Skin_Friction_Coefficient': [
            2.51354891e-03,
            -2.06068039e-05,
            6.91914513e-05
        ],
    }

    integral_cell_res = {
        'AREA': 1.555723245815826
    }

    aero_coeff = AeroCoeff(
        CL=0.2640808193,
        CD=0.0143852168,
        CSF=0.0132942137,
        CFx=0.0002671397,
        CFy=0.0132942137,
        CFz=0.2644725394,
        CMx=0.2163125907,
        CMy=-0.08382616185,
        CMz=0.016735067067,
    )

    def test_triangulate(self):

        grid = self.grid.copy()
        grid.triangulate()

        self.assertTrue(np.all(grid.cell_type == 5))

        aero_coeff = self.grid.compute_aero_coeff()

        for i in range(6):
            self.assertTrue(
                np.allclose(self.aero_coeff[i], aero_coeff[i], rtol=10*self.rtol),
                'Computed {} is {}, expecting {}'
                .format(aero_coeff._fields[i], aero_coeff[i], self.aero_coeff[i])
            )


class TestONERAInv(TestRAE2822RANS):

    filename = os.path.join(os.path.dirname(__file__), 'oneram6_inv_tetra.vtk')

    rtol = 1e-4

    angle_of_attack = 3.06
    ref_length = 0.64607
    ref_area = 0.7532
    ref_semispan = 1.1963
    ref_origin = [0.25, 0., 0.]

    integral_pts_res = {
        'Pressure_Coefficient': -0.3587423438026275,
    }

    integral_cell_res = {
        'AREA': 1.556044299015
    }

    aero_coeff = AeroCoeff(
        CL=0.2831954389,
        CD=0.01188402067,
        CSF=0.01457329742,
        CFx=-0.003250374846,
        CFy=0.01457329742,
        CFz=0.2834260426,
        CMx=0.2306559093,
        CMy=-0.097605512,
        CMz=0.0210972655,
    )


if __name__ == '__main__':
    unittest.main()
