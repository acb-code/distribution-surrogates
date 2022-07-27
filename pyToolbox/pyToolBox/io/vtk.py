import os
import numpy as np
import vtk
import vtk.util.numpy_support as vn

from ..geom import GridData

CELLTYPE_ENUM = {
        3: {'name': 'LINE', 'npts': 2},
        5: {'name': 'TRIANGLE', 'npts': 3},
        9: {'name': 'QUAD', 'npts': 4},
        10: {'name': 'TETRA', 'npts': 4},
        12: {'name': 'HEXAHEDRON', 'npts': 8},
        13: {'name': 'WEDGE', 'npts': 6},
        14: {'name': 'PYRAMID', 'npts': 5},
    }


def read_vtk_file(filename: str, exclude_fields: [None, list] = None, vtk_format: str = 'auto') -> GridData:

    if not os.path.isfile(filename):
        raise FileNotFoundError

    assert vtk_format in ('auto', 'legacy', 'xml')

    if vtk_format == 'auto':
        ext = os.path.splitext(filename)[-1]
        if ext == '.vtk':
            _vtk_format = 'legacy'
        elif ext == '.vtu':
            _vtk_format = 'xml'
        else:
            raise ValueError
    else:
        _vtk_format = vtk_format

    if exclude_fields is None:
        exclude_fields = []
    else:
        assert isinstance(exclude_fields, (list, tuple, set))

    if _vtk_format == 'legacy':
        vtk_reader = vtk.vtkUnstructuredGridReader()
        vtk_reader.ReadAllFieldsOn()
        vtk_reader.ReadAllVectorsOn()
        vtk_reader.ReadAllScalarsOn()
    else:
        vtk_reader = vtk.vtkXMLUnstructuredGridReader()

    vtk_reader.SetFileName(filename)
    vtk_reader.Update()

    vtk_object = vtk_reader.GetOutput()

    points = vn.vtk_to_numpy(vtk_object.GetPoints().GetData())
    point_ids = vn.vtk_to_numpy(vtk_object.GetCells().GetData())

    if vtk_object.IsHomogeneous():
        cell_type = vtk_object.GetCellType(0)
    else:
        raise NotImplementedError

    cell_npts = CELLTYPE_ENUM[cell_type]['npts']
    point_ids = point_ids.reshape(-1, cell_npts + 1)

    data_out = GridData()
    data_out.set_points(points)
    data_out.set_cells(point_ids[:, 1:], cell_type)

    vtk_point_data = vtk_object.GetPointData()
    num_point_arrays = vtk_point_data.GetNumberOfArrays()
    for i in range(num_point_arrays):
        arr_name = vtk_point_data.GetArrayName(i)
        if arr_name in exclude_fields:
            continue
        else:
            data_out.add_point_array(arr_name, vn.vtk_to_numpy(vtk_point_data.GetArray(i)))

    vtk_cell_data = vtk_object.GetCellData()
    num_cell_arrays = vtk_cell_data.GetNumberOfArrays()
    for i in range(num_cell_arrays):
        arr_name = vtk_cell_data.GetArrayName(i)
        if arr_name in exclude_fields:
            continue
        else:
            data_out.add_cell_array(arr_name, vn.vtk_to_numpy(vtk_cell_data.GetArray(i)))

    return data_out


def write_vtk_file(filename: str, data: GridData,
                   overwrite: bool = True, binary: bool = True,
                   xml: bool = True) -> None:

    if os.path.isfile(filename) and not overwrite:
        raise FileExistsError

    vtk_object = vtk.vtkUnstructuredGrid()

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vn.numpy_to_vtk(data.points))
    vtk_object.SetPoints(vtk_points)

    cell_npts = CELLTYPE_ENUM[data.cell_type]['npts']
    num_cells = data.num_cells
    point_ids = np.empty((num_cells, cell_npts + 1), dtype=np.int64)
    point_ids[:, 0] = cell_npts
    point_ids[:, 1:] = data.point_ids

    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(data.num_cells, vn.numpy_to_vtkIdTypeArray(point_ids.ravel()))
    vtk_object.SetCells(data.cell_type, vtk_cells)

    point_data = data.point_data
    vtk_point_data = vtk_object.GetPointData()
    for arr_name, arr_data in point_data.items():
        vtk_arr = vn.numpy_to_vtk(arr_data)
        vtk_arr.SetName(arr_name)
        vtk_point_data.AddArray(vtk_arr)

    cell_data = data.cell_data
    vtk_cell_data = vtk_object.GetCellData()
    for arr_name, arr_data in cell_data.items():
        vtk_arr = vn.numpy_to_vtk(arr_data)
        vtk_arr.SetName(arr_name)
        vtk_cell_data.AddArray(vtk_arr)

    _filename = os.path.splitext(filename)[0]
    _filename += '.vtu' if xml else '.vtk'

    if xml:
        vtk_writer = vtk.vtkXMLUnstructuredGridWriter()
        if binary:
            vtk_writer.SetDataModeToAppended()
        else:
            vtk_writer.SetDataModeToAscii()

    else:
        vtk_writer = vtk.vtkUnstructuredGridWriter()
        vtk_writer.SetHeader('VTK Output')
        if binary:
            vtk_writer.SetFileTypeToBinary()
        else:
            vtk_writer.SetFileTypeToASCII()

    vtk_writer.SetFileName(_filename)
    vtk_writer.SetInputData(vtk_object)
    vtk_writer.Write()
