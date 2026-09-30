"""FVM solver for the Reynolds equation on a rectangular tilted pad.

Solves d/dX(phi_x dP/dX) + d/dY(phi_y dP/dY) = d(phi_s)/dX with P = 0 on the
edges. For a smooth film phi_x = phi_y = H**3 and phi_s = H, which are the
defaults. Passing flow factors from homogenised.table instead gives the
homogenised equation. 
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from . import assembly, boundary, film


def solve(
    # Input data
    length=1e-3,
    width=1e-3,
    h_taper=5e-6,
    h_min=20e-6,
    # Grid size
    x_nodes=50,
    y_nodes=50,
    film_thickness=film.cartesian,
    # Flow factors
    phi_x=lambda H: H**3,
    phi_y=lambda H: H**3,
    phi_s=lambda H: H,
):
    H_taper = h_taper / h_min

    # Grid
    X_vector = np.linspace(0, 1, x_nodes)
    Y_vector = np.linspace(0, width / length, y_nodes)
    X, Y = np.meshgrid(X_vector, Y_vector)

    # Dimensionless increments
    dX = 1 / (x_nodes - 1)
    dY = (width / length) / (y_nodes - 1)

    # Dimensionless height function
    H = film_thickness(length, width, h_taper, h_min)

    # Height at intermediate points
    H_east, H_west, H_north, H_south = film.faces(H, X, Y, dX, dY)

    # Coefficients for FVM formulation
    # a_P*P_P = a_E*P_E+...+a_S*P_S+source
    a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
        phi_x(H_east), phi_x(H_west), phi_y(H_north), phi_y(H_south), dX, dY
    )
    source = -assembly.build_source(
        phi_s(H_east), phi_s(H_west), phi_s(H_north), phi_s(H_south), dX, dY
    )

    # Pressure is zero on the edges
    fixed = boundary.all_edges(x_nodes, y_nodes)
    rhs = assembly.identity_rhs(source, fixed)

    # Building the system
    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, x_nodes, y_nodes, fixed
    )

    # Solving
    pressure = spsolve(A.tocsr(), rhs).reshape(y_nodes, x_nodes)

    return pressure, X_vector, Y_vector, H_taper
