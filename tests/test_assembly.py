"""Structural checks on the assembled matrix.

Takes no inputs of its own, each test assembles a small matrix directly.
Fails if a node gets connected to the wrong neighbour, or if a boundary row is
not reduced to the identity.

The links are checked with no boundary nodes at all. With the usual Dirichlet
edges the edge rows carry no links anyway, so a wrong link there would be masked
away before a test could see it.
"""

import numpy as np
from scipy.sparse import csr_array

from reynolds import assembly, boundary

N_COLS, N_ROWS = 5, 4
SIZE = N_COLS * N_ROWS


def unit_coefficients():
    """Five arrays of ones, so the structure is visible without coefficient noise."""
    ones = np.ones(SIZE)
    return ones.copy(), ones.copy(), ones.copy(), ones.copy(), 4 * ones.copy()


def test_east_west_links_never_cross_grid_rows():
    """Nodes are numbered row major, so node p and node p + 1 are neighbours in
    the numbering even when p sits in the last column, where they are really on
    opposite edges of the grid. Those two must not end up linked."""
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
    """A boundary row must read pressure * 1 = 0, so it carries the diagonal and
    nothing else."""
    a_east, a_west, a_north, a_south, a_P = unit_coefficients()
    is_boundary = boundary.all_edges(N_COLS, N_ROWS)
    a_P[is_boundary] = 1.0

    A = csr_array(
        assembly.five_point(
            a_east, a_west, a_north, a_south, a_P, N_COLS, N_ROWS, is_boundary
        )
    )

    for node in np.flatnonzero(is_boundary):
        assert A[[node]].nnz == 1
        assert A[node, node] == 1.0
