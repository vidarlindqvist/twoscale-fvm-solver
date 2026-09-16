"""FVM solver for the Reynolds equation on a rectangular tilted pad.

Takes the pad length and width, the film taper and the grid size.
Returns the dimensionless pressure field, the X and Y grid vectors and delta_H.
"""

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


def solve(
    # Input data
    l=1e-3,
    b=1e-3,
    delta_h=5e-6,
    h0=20e-6,
    # Grid size
    x_nodes=50,
    y_nodes=50,
):
    delta_H = delta_h / h0

    # Grid
    X_vector = np.linspace(0, 1, x_nodes)
    Y_vector = np.linspace(b/l, 0, y_nodes)
    X, Y = np.meshgrid(X_vector, Y_vector)

    # Dimensionless increments
    delta_X = 1 / (x_nodes - 1)
    delta_Y = (b / l) / (y_nodes - 1)

    # Real height function
    def h(x, y):
        return h0 + delta_h * (l - x) / l

    # Dimensionless height function
    def H(x, y):
        return h(x * l, y * b) / h0

    # Height at intermediate points
    H_east = H(X + delta_X / 2, Y)
    H_west = H(X - delta_X / 2, Y)
    H_north = H(X, Y + delta_Y / 2)
    H_south = H(X, Y - delta_Y / 2)

    # Coefficients for FVM formulation
    # a_P*P_P = a_E+...+a_S+C_P
    a_east = 1 / delta_X**2 * H_east.flatten() ** 3
    a_west = 1 / delta_X**2 * H_west.flatten() ** 3
    a_north = 1 / delta_Y**2 * H_north.flatten() ** 3
    a_south = 1 / delta_Y**2 * H_south.flatten() ** 3
    a_P = a_east + a_west + a_north + a_south
    C_P = -((H_east - H_west) / delta_X).flatten()

    # Finding the boundaries. 1 if it is on the boundary, 0 if not.
    is_boundary = np.zeros((y_nodes, x_nodes), dtype = bool)
    is_boundary[0,:] = True
    is_boundary[:,0] = True
    is_boundary[-1,:] = True
    is_boundary[:,-1] = True

    is_boundary = is_boundary.flatten()

    # Building and formatting the diagonals for spdiags
    east_diagonal = -np.append([0], (a_east * ~is_boundary)[:-1])
    west_diagonal = -np.append((a_west * ~is_boundary)[1:], [0])
    north_diagonal = -np.append(np.zeros((x_nodes, )), (a_north * ~is_boundary)[:-x_nodes])
    south_diagonal = -np.append((a_south * ~is_boundary)[x_nodes:], np.zeros((x_nodes, )))

    # The equation will be pressure * a_P = 0 on the boundary, forcing pressure = 0
    a_P[is_boundary] = 1.0
    C_P[is_boundary] = 0.0

    # Building the system
    diagonals = [a_P, east_diagonal, west_diagonal, north_diagonal, south_diagonal]
    A = spdiags(diagonals, [0, 1, -1, x_nodes, -x_nodes], [x_nodes * y_nodes, x_nodes * y_nodes])

    # Solving
    pressure = spsolve(A.tocsr(), C_P).reshape(y_nodes, x_nodes)

    return pressure, X_vector, Y_vector, delta_H

