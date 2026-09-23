"""FVM solver for the Reynolds equation on a rectangular tilted pad.

Takes the pad length and width, the film taper, the grid size and optionally a
film thickness callable overriding the default tilted pad.
Returns the dimensionless pressure field, the X and Y grid vectors and delta_H.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from . import assembly, film, boundary


def solve(
    # Input data
    l=1e-3,
    b=1e-3,
    delta_h=5e-6,
    h0=20e-6,
    # Grid size
    x_nodes=50,
    y_nodes=50,
    film_thickness=film.cartesian,
):
    delta_H = delta_h / h0

    # Grid
    X_vector = np.linspace(0, 1, x_nodes)
    Y_vector = np.linspace(b / l, 0, y_nodes)
    X, Y = np.meshgrid(X_vector, Y_vector)

    # Dimensionless increments
    delta_X = 1 / (x_nodes - 1)
    delta_Y = (b / l) / (y_nodes - 1)

    # Dimensionless height function
    H = film_thickness(l, b, delta_h, h0)

    # Height at intermediate points
    H_east, H_west, H_north, H_south = film.faces(H, X, Y, delta_X, delta_Y)

    # Coefficients for FVM formulation
    # a_P*P_P = a_E+...+a_S+C_P
    a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficiants(
        H_east, H_west, H_north, H_south, delta_X, delta_Y)
    
    source = assembly.build_source(H_east, H_west, H_north, H_south, delta_X, delta_Y, k=1, e=[1,0])
    
    pinned = boundary.all_edges(x_nodes, y_nodes)
    rhs = assembly.identity_rhs(source, pinned)


    # Building the system
    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, x_nodes, y_nodes,
        pinned, periodic=False
    )

    # Solving
    pressure = spsolve(A.tocsr(), rhs).reshape(y_nodes, x_nodes)
    return pressure, X_vector, Y_vector, delta_H
