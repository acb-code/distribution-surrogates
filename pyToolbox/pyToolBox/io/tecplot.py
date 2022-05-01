import os
import numpy as np

from ..geom import GridData


def write_tecplot_ascii(filename: str, data: GridData,
                        overwrite: bool = True,
                        header: str = 'Tecplot output') -> None:

    if os.path.isfile(filename) and not overwrite:
        raise FileExistsError

    if data.cell_type == 3:
        ztype = 'FELINESEG'
    elif data.cell_type == 5:
        ztype = 'FETRIANGLE'
    elif data.cell_type == 9:
        ztype = 'FEQUADRILATERAL'
    else:
        raise NotImplementedError

    with open(filename, 'w') as out:
        out.write('TITLE=\"{}\"\n'.format(header))
        out.write('VARIABLES=\n')

        coord_list = ['x', 'y', 'z']

        point_list = []
        for k, arr in data.point_data.items():
            if arr.ndim == 1:
                point_list += [k]
            else:
                point_list += ['{}_{}'.format(k, d) for d in ('x', 'y', 'z')]

        cell_list = []
        for k, arr in data.cell_data.items():
            if arr.ndim == 1:
                cell_list += [k]
            else:
                cell_list += ['{}_{}'.format(k, d) for d in ('x', 'y', 'z')]

        var_list = coord_list + point_list + cell_list
        out.write(', '.join('\"{}\"'.format(var) for var in var_list) + '\n')
        out.write('ZONE NODES={:d}, ELEMENTS={:d}, DATAPACKING=BLOCK, ZONETYPE={}\n'
                  .format(data.num_points, data.num_cells, ztype))

        if cell_list:
            istart = len(coord_list + point_list) + 1
            iend = istart + len(cell_list) - 1
            if istart == iend:
                out.write('VARLOCATION=([{}]=CELLCENTERED)\n'.format(istart))
            else:
                out.write('VARLOCATION=([{}-{}]=CELLCENTERED)\n'.format(istart, iend))

        for j in range(3):
            _write_grid_data(out, data.points[:, j])

        for _, arr in data.point_data.items():
            if arr.ndim == 1:
                _write_grid_data(out, arr)
            else:
                for j in range(3):
                    _write_grid_data(out, arr[:, j])

        for _, arr in data.cell_data.items():
            if arr.ndim == 1:
                _write_grid_data(out, arr)
            else:
                for j in range(3):
                    _write_grid_data(out, arr[:, j])

        _write_grid_connectivity(out, data.point_ids)


def _write_grid_data(fid, arr, line_size=10):

    _arr = arr.ravel()
    precision = np.finfo(_arr.dtype).precision

    str_f = '{: .' + str(precision) + 'e}'

    i = 0
    di = line_size

    while i < _arr.size:
        line_str = '\t'.join(str_f.format(nb) for nb in _arr[i:i + di]) + '\n'
        fid.write(line_str)
        i += di


def _write_grid_connectivity(fid, point_ids):

    for cell in point_ids:
        line_str = ' '.join('{:d}'.format(c) for c in (cell + 1)) + '\n'
        fid.write(line_str)
