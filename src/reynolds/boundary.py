"""Which nodes sit on a boundary.

Takes the grid shape.
Returns a flat boolean mask, ready to pass as the is_boundary argument of
assembly.five_point.

Both solvers impose the same condition on every edge node, so the mask is built
here rather than written out twice.
"""

import numpy as np


def all_edges(n_cols, n_rows):
    """Every node on the four edges of the grid, for a Dirichlet condition."""
    edges = np.zeros((n_rows, n_cols), dtype=bool)
    edges[0, :] = True
    edges[:, 0] = True
    edges[-1, :] = True
    edges[:, -1] = True
    return edges.flatten()
