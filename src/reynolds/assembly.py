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
    fixed,
    periodic=False,
):
    size = n_cols * n_rows
    node = np.arange(size)
    grid_row = node // n_cols
    grid_col = node % n_cols

    a_P = a_P.copy()
    a_P[fixed] = 1.0

    rows = [node]
    cols = [node]
    values = [a_P]

    coefficients = (a_east, a_west, a_north, a_south)
    for coefficient, (row_step, col_step) in zip(coefficients, _DIRECTIONS):
        neighbour, exists = _neighbour(
            grid_row, grid_col, row_step, col_step, n_cols, n_rows, periodic
        )
        linked = exists & ~fixed
        rows.append(node[linked])
        cols.append(neighbour[linked])
        values.append(-coefficient[linked])

    return coo_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
    )


def identity_rhs(source, fixed, value=0.0):
    rhs = np.asarray(source, dtype=float).copy()
    rhs[fixed] = value

    return rhs


def cartesian_coefficients(K_east, K_west, K_north, K_south, dX1, dX2):
    a_east = dX2 / dX1 * K_east.flatten()
    a_west = dX2 / dX1 * K_west.flatten()
    a_north = dX1 / dX2 * K_north.flatten()
    a_south = dX1 / dX2 * K_south.flatten()
    a_P = a_east + a_west + a_north + a_south

    return a_east, a_west, a_north, a_south, a_P


def build_source(Q_east, Q_west, Q_north, Q_south, dX1, dX2, e=(1, 0)):
    source = (e[0] * dX2 * (Q_east - Q_west) + e[1] * dX1 * (Q_north - Q_south)).flatten()

    return source

def cavitation_coefficients(H_east, H_west, dX2):
    b_east = dX2 / 2 * H_east.flatten()
    b_west = - dX2 / 2 * H_west.flatten()
    b_P = b_east + b_west
    
    return b_east, b_west, b_P



