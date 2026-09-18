"""Film thickness definitions for the tilted pad geometries.

Takes the pad geometry and the film taper.
Returns a callable giving the dimensionless film thickness on the dimensionless grid.

faces() evaluates any of those callables at the four cell faces, which is the
pattern every solver needs before forming its coefficients.
"""

import numpy as np


def cartesian(l, b, delta_h, h0):
    # Real height function
    def h(x, y):
        return h0 + delta_h * (l - x) / l

    # Dimensionless height function
    def H(x, y):
        return h(x * l, y * b) / h0

    return H


def polar(Ri, Ro, delta_h, h0, theta2):
    # Real height function
    def h(r, theta):
        return h0 + delta_h * r * np.sin(theta2 - theta) / (
            (Ri + Ro) / 2 * np.sin(theta2)
        )

    # Dimensionless height function
    def H(r, theta):
        return h(r * Ro, theta) / h0

    return H


def faces(H, first, second, d_first, d_second):
    """Film thickness at the east, west, north and south faces of every cell."""
    return (
        H(first + d_first / 2, second),
        H(first - d_first / 2, second),
        H(first, second + d_second / 2),
        H(first, second - d_second / 2),
    )
