"""Sparse matrix assembly for the five point FVM stencil.

Takes the face coefficients, the diagonal, the grid shape and which rows are
Dirichlet.
Returns a COO sparse matrix, built from explicit (row, column, value) triplets.

Both the cartesian and the polar solver assemble the same operator and differ
only in what the coefficients mean, so the placement of those coefficients into
a matrix lives here rather than being written out twice.
"""

import numpy as np
from scipy.sparse import coo_array


def five_point(a_east, a_west, a_north, a_south, a_P, n_cols, n_rows, is_boundary):
    """Assemble a_P*P_P = a_E*P_E + a_W*P_W + a_N*P_N + a_S*P_S on a structured grid.

    Rows flagged in is_boundary are emitted as identity rows, which imposes
    the Dirichlet condition directly.
    """
    size = n_cols * n_rows
    node = np.arange(size)
    grid_row = node // n_cols
    grid_col = node % n_cols
    interior = ~is_boundary

    # offset to the neighbour, its coefficient, and whether that neighbour exists.
    neighbours = (
        (1, a_east, grid_col + 1 < n_cols),
        (-1, a_west, grid_col - 1 >= 0),
        (n_cols, a_north, grid_row + 1 < n_rows),
        (-n_cols, a_south, grid_row - 1 >= 0),
    )

    rows = [node]
    cols = [node]
    values = [a_P]

    for offset, coefficient, neighbour_exists in neighbours:
        connected = neighbour_exists & interior
        rows.append(node[connected])
        cols.append(node[connected] + offset)
        values.append(-coefficient[connected])

    return coo_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
    )
