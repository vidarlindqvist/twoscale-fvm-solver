"""Which nodes have a prescribed value.

Takes the grid shape.
Returns a flat boolean mask, ready to pass as the fixed argument of
assembly.five_point.

The pad solvers fix every edge node (Dirichlet, ambient pressure). The periodic
cell solver has no edges, so it fixes a single node instead.
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


def pin(size, node=0):
    """One node, for a periodic problem whose solution is only defined up to a
    constant. Fixing it removes that constant and nothing else."""
    pinned = np.zeros(size, dtype=bool)
    pinned[node] = True
    return pinned
