import numpy as np
from scipy.sparse.linalg import spsolve

from . import assembly, boundary, film


def solve(
    length=1e-3,
    width=1e-3,
    h_taper=5e-6,
    h_min=20e-6,
    x_nodes=50,
    y_nodes=50,
    film_thickness=film.cartesian,
):
    H_taper = h_taper / h_min
    X_vector = np.linspace(0, 1, x_nodes)
    Y_vector = np.linspace(0, width / length, y_nodes)

    X, Y = np.meshgrid(X_vector, Y_vector)

    dX = 1 / (x_nodes - 1)
    dY = (width / length) / (y_nodes - 1)

    H = film_thickness(length, width, h_taper, h_min)

    H_east, H_west, H_north, H_south = film.faces(H, X, Y, dX, dY)

    a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
        H_east**3, H_west**3, H_north**3, H_south**3, dX, dY
    )

    source = -assembly.build_source(H_east, H_west, H_north, H_south, dX, dY, e=(1, 0))

    fixed = boundary.all_edges(x_nodes, y_nodes)
    rhs = assembly.identity_rhs(source, fixed)

    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, x_nodes, y_nodes, fixed
    )

    pressure = spsolve(A.tocsr(), rhs).reshape(y_nodes, x_nodes)

    return pressure, X_vector, Y_vector, H_taper
