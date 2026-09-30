"""Structure of the five point matrix, on a small grid of unit coefficients."""

import numpy as np
from scipy.sparse import csr_array

from reynolds import assembly, boundary

N_COLS, N_ROWS = 5, 4
SIZE = N_COLS * N_ROWS


def unit_coefficients():
    ones = np.ones(SIZE)
    return ones.copy(), ones.copy(), ones.copy(), ones.copy(), 4 * ones.copy()


def test_east_west_links_never_cross_grid_rows():
    # Row major numbering: the last node in a row is followed by the first
    # node of the next row, which is not a neighbour. No fixed nodes here,
    # since fixed rows have no links and would hide a wrong one.
    a_east, a_west, a_north, a_south, a_P = unit_coefficients()
    no_boundary = np.zeros(SIZE, dtype=bool)

    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, N_COLS, N_ROWS, no_boundary
    ).tocoo()

    crossing = [
        (int(row), int(col))
        for row, col in zip(A.row, A.col)
        if abs(int(row) - int(col)) == 1 and int(row) // N_COLS != int(col) // N_COLS
    ]

    assert crossing == []


def test_boundary_rows_are_emitted_as_identity():
    a_east, a_west, a_north, a_south, a_P = unit_coefficients()
    fixed = boundary.all_edges(N_COLS, N_ROWS)
    a_P[fixed] = 1.0

    A = csr_array(
        assembly.five_point(
            a_east, a_west, a_north, a_south, a_P, N_COLS, N_ROWS, fixed
        )
    )

    for node in np.flatnonzero(fixed):
        assert A[[node]].nnz == 1
        assert A[node, node] == 1.0
