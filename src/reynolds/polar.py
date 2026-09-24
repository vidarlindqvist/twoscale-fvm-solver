"""FVM solver for the Reynolds equation on an annular sector tilted pad.

Takes the inner and outer radius, the smallest gap, the film taper, the grid
size, the pad angle and optionally a film thickness callable overriding the
default tilted pad.
Returns the dimensionless pressure field, the R and theta grid vectors and theta2.

Scaling: R = r / Ro, H = h / h_min and P = p * h_min**2 / (6 * mu * omega * Ro**2).
The solver then solves d/dR(R H**3 dP/dR) + 1/R d/dtheta(H**3 dP/dtheta) = R dH/dtheta,
which is the polar Reynolds equation multiplied by R, integrated over dR dtheta.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from . import assembly, boundary, film


def solve(
    # Input data
    Ri=499e-3,
    Ro=500e-3,
    h_min=20e-6,
    h_taper=5e-6,
    # Grid size
    r_nodes=50,
    theta_nodes=50,
    theta1=0,
    theta2=55 * np.pi / 180,
    film_thickness=film.polar,
):
    # (Ro-Ri) = Ri * theta2
    # Grid
    R_vector = np.linspace(Ri / Ro, 1, r_nodes)
    theta_vector = np.linspace(theta1, theta2, theta_nodes)
    R, theta = np.meshgrid(R_vector, theta_vector)

    # Dimensionless increments
    dR = (1 - Ri / Ro) / (r_nodes - 1)
    dtheta = (theta2 - theta1) / (theta_nodes - 1)

    # Intermediate points
    R_east = R + dR / 2
    R_west = R - dR / 2
    theta_north = theta + dtheta / 2
    theta_south = theta - dtheta / 2

    # Dimensionless height function
    H = film_thickness(Ri, Ro, h_taper, h_min, theta2)

    # Height at intermediate points
    H_east, H_west, H_north, H_south = film.faces(H, R, theta, dR, dtheta)

    # Coefficients for FVM formulation
    # a_P*P_P = a_E*P_E+...+a_S*P_S+source
    a_east = (dtheta * R_east / dR * H_east**3).flatten()
    a_west = (dtheta * R_west / dR * H_west**3).flatten()
    a_north = (dR / (R * dtheta) * H_north**3).flatten()
    a_south = (dR / (R * dtheta) * H_south**3).flatten()
    a_P = a_east + a_west + a_north + a_south

    # source = int R (H(R, theta_south) - H(R, theta_north)) dR over the cell.
    # Note the factor R: the equation was multiplied by R. Simpson's rule over
    # [R_west, R_east] is exact when H is linear in R, as for the tilted pad.
    def shear(R_k):
        return R_k * (H(R_k, theta_south) - H(R_k, theta_north))

    source = (dR / 6 * (shear(R_west) + 4 * shear(R) + shear(R_east))).flatten()

    # Finding the boundaries. 1 if it is on the boundary, 0 if not.
    fixed = boundary.all_edges(r_nodes, theta_nodes)

    # The equation will be pressure * 1 = 0 on the boundary, forcing pressure = 0
    rhs = assembly.identity_rhs(source, fixed)

    # Building the system
    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, r_nodes, theta_nodes, fixed
    )

    # Solving
    pressure = spsolve(A.tocsr(), rhs).reshape(theta_nodes, r_nodes)

    return pressure, R_vector, theta_vector, theta2
