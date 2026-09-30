"""Masks of the nodes with a prescribed value, passed as fixed to assembly.five_point."""

import numpy as np


def all_edges(n_cols, n_rows):
    # Every node on the four edges, for the pad
    edges = np.zeros((n_rows, n_cols), dtype=bool)
    edges[0, :] = True
    edges[:, 0] = True
    edges[-1, :] = True
    edges[:, -1] = True
    return edges.flatten()


def pin(size, node=0):
    # A single node, for the periodic cell where the solution is only
    # defined up to a constant
    pinned = np.zeros(size, dtype=bool)
    pinned[node] = True
    return pinned
