import numpy as np
from scipy.sparse import coo_array

_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _neighbour(grid_row, grid_col, row_step, col_step, n_cols, n_rows, periodic):
    row = grid_row + row_step
    col = grid_col + col_step
    node = (row % n_rows) * n_cols + (col % n_cols)
    if periodic:
        return node, np.ones_like(node, dtype=bool)

    return node, (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)


def five_point(
    a_east,
    a_west,
    a_north,
    a_south,
    a_P,
    n_cols,
    n_rows,
    identity_rows,
    periodic=False,
):
    size = n_cols * n_rows
    node = np.arange(size)
    grid_row = node // n_cols
    grid_col = node % n_cols

    a_P = a_P.copy()
    a_P[identity_rows] = 1.0

    rows = [node]
    cols = [node]
    values = [a_P]

    coefficients = (a_east, a_west, a_north, a_south)
    for coefficient, (row_step, col_step) in zip(coefficients, _DIRECTIONS):
        neighbour, exists = _neighbour(
            grid_row, grid_col, row_step, col_step, n_cols, n_rows, periodic
        )
        linked = exists & ~identity_rows
        rows.append(node[linked])
        cols.append(neighbour[linked])
        values.append(-coefficient[linked])
    
    return coo_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
    )


def identity_rhs(C_P, identity_rows, value=0.0):
    rhs = np.asarray(C_P, dtype=float).copy()
    rhs[identity_rows] = value
    return rhs

def cartesian_coefficiants(H_east, H_west, H_north, H_south, delta_x, delta_y):
    
    a_east = delta_y / delta_x * H_east.flatten() ** 3
    a_west = delta_y / delta_x * H_west.flatten() ** 3
    a_north = delta_x / delta_y * H_north.flatten() ** 3
    a_south = delta_x / delta_y * H_south.flatten() ** 3
    a_P = a_east + a_west + a_north + a_south
    
    return a_east, a_west, a_north, a_south, a_P

def build_source(H_east, H_west, H_north, H_south, delta_x, delta_y, k, e):

    source = ( e[0] * delta_y * (H_east ** k - H_west ** k) + e[1] * delta_x * (H_north ** k - H_south ** k) ).flatten()
    
    return source
    