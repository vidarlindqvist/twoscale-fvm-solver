import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


def solve(
    # Input data
    Ri=299e-3,
    Ro=300e-3,
    h0=20e-6,
    delta_h=5e-6,
    # Grid size
    r_nodes=50,
    theta_nodes=50,
):
    theta1 = 0
    theta2 = (Ro - Ri) / Ri # 55 *np.pi / 180

    # Grid
    R_vector = np.linspace(Ri/Ro, 1, r_nodes)
    theta_vector = np.linspace(theta1, theta2, theta_nodes)
    R, theta = np.meshgrid(R_vector, theta_vector)

    # Dimensionless increments
    delta_R = (1 - Ri / Ro) / (r_nodes - 1)
    delta_theta = (theta2 - theta1) / (theta_nodes - 1)

    # Intermediate points
    R_east = R + delta_R / 2
    R_west = R - delta_R / 2
    theta_north = theta + delta_theta / 2
    theta_south = theta - delta_theta / 2

    # Real height function
    def h(r, theta):
        return h0 + delta_h * r * np.sin(theta2 - theta) / ( (Ri + Ro) / 2 * np.sin(theta2))
    # Dimensionless height function
    def H(r, theta):
        return h(r * Ro, theta) / h0

    # Height at intermediate points
    H_east = H(R_east, theta)
    H_west = H(R_west, theta)
    H_north = H(R, theta_north)
    H_south = H(R, theta_south)

    # Coefficients for FVM formulation
    # a_P*P_P = a_E+...+a_S+C_P
    a_east = ( delta_theta * R_east / delta_R * H_east  ** 3).flatten()
    a_west = ( delta_theta * R_west / delta_R * H_west ** 3 ).flatten()
    a_north = ( delta_R / (R * delta_theta) * H_north ** 3 ).flatten()
    a_south = ( delta_R / (R * delta_theta) * H_south ** 3 ).flatten()
    a_P = a_east + a_west + a_north + a_south
        # C_P is correct. We are exploiting the way the integral computes to comptact the line.
        # # C_P = int H(theta1,R)-H(theta2,R) dR
    C_P = ( (H(1, theta_south) - H(1, theta_north)) / 2 * (R_east ** 2 - R_west ** 2) ).flatten()

    # Finding the boundaries. 1 if it is on the boundary, 0 if not.
    is_boundary = np.zeros((theta_nodes, r_nodes), dtype = bool)
    is_boundary[0,:] = True
    is_boundary[:,0] = True
    is_boundary[-1,:] = True
    is_boundary[:,-1] = True

    is_boundary = is_boundary.flatten()

    # Building and formatting the diagonals for spdiags
    east_diagonal = -np.append([0], (a_east * ~is_boundary)[:-1])
    west_diagonal = -np.append((a_west * ~is_boundary)[1:], [0])
    north_diagonal = -np.append(np.zeros((r_nodes, )), (a_north * ~is_boundary)[:-r_nodes])
    south_diagonal = -np.append((a_south * ~is_boundary)[r_nodes:], np.zeros((r_nodes, )))

    # The equation will be pressure * a_P = 0 on the boundary, forcing pressure = 0
    a_P[is_boundary] = 1.0
    C_P[is_boundary] = 0.0

    # Building the system
    diagonals = [a_P, east_diagonal, west_diagonal, north_diagonal, south_diagonal]
    A = spdiags(diagonals, [0, 1, -1, r_nodes, -r_nodes], [r_nodes * theta_nodes, r_nodes * theta_nodes])

    # Solving
    pressure = spsolve(A.tocsr(), C_P).reshape(theta_nodes, r_nodes)

    return pressure, R_vector, theta_vector, theta2


#plt.subplot(projection="3d").plot_surface(*np.indices(pressure.shape), pressure, cmap="viridis")
