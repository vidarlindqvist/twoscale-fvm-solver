"""Flow factors for the homogenised Reynolds equation.

The roughness is given on the unit cell 0 <= y1, y2 < 1 and the local film is
H = alpha + H_r(y1, y2), where alpha is the smooth pad film seen from the cell.
The cell problems are solved with the same FVM as the pad, but with
periodic boundaries, and averaged into the coefficients.
Repeating this for a range of alpha gives a table that the pad solver
interpolates in. 
"""

import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import splu

from . import assembly, boundary, film

# Power of H in the source term, its direction and its sign.
CELL_PROBLEMS = {
    "psi1": (1, (1, 0), -1),
    "psi2": (1, (0, 1), -1),
    "chi1": (3, (1, 0), 1),
    "chi2": (3, (0, 1), 1),
}


def cell_film(roughness, alpha, n):
    # n nodes per direction, the node after the last one is the first one again
    y_vector = np.arange(n) / n
    dy = 1 / n
    Y1, Y2 = np.meshgrid(y_vector, y_vector)

    def H(y1, y2):
        return alpha + roughness(y1, y2)

    return y_vector, dy, film.faces(H, Y1, Y2, dy, dy)


def solve_cell(roughness, alpha, problems=("chi1", "chi2", "psi1"), n=64):
    y_vector, dy, (H_east, H_west, H_north, H_south) = cell_film(roughness, alpha, n)

    a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
        H_east**3, H_west**3, H_north**3, H_south**3, dy, dy
    )

    # With periodic boundaries the solution is only defined up to a constant,
    # so one node is fixed to zero. The matrix is the same for every problem.
    fixed = boundary.pin(n * n)
    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, n, n, fixed, periodic=True
    )
    lu = splu(A.tocsc())

    solutions = {}
    for problem in problems:
        k, e, sign = CELL_PROBLEMS[problem]
        source = sign * assembly.build_source(
            H_east**k, H_west**k, H_north**k, H_south**k, dy, dy, e
        )
        rhs = assembly.identity_rhs(source, fixed)
        solutions[problem] = lu.solve(rhs).reshape(n, n)

    return solutions, y_vector


def flow_factors(roughness, alpha, n=64):
    solutions, _ = solve_cell(roughness, alpha, n=n)
    chi1, chi2, psi1 = solutions["chi1"], solutions["chi2"], solutions["psi1"]
    _, dy, (H_east, _, H_north, _) = cell_film(roughness, alpha, n)

    # Derivatives across the east (axis 1) and north (axis 0) faces
    def d_east(u):
        return (np.roll(u, -1, axis=1) - u) / dy

    def d_north(u):
        return (np.roll(u, -1, axis=0) - u) / dy

    # Cell averages of the fluxes through the faces. 
    phi_x = np.mean(H_east**3 * (1 + d_east(chi1)))  # h0**3 a11, (5.324a)
    phi_y = np.mean(H_north**3 * (1 + d_north(chi2)))  # h0**3 a22, (5.324d)
    phi_s = np.mean(H_east - H_east**3 * d_east(psi1))  # h0 b11, (5.325a)

    # Cross terms, zero for roughness that is symmetric in y1 and y2
    phi_xy = np.mean(H_east**3 * d_east(chi2))
    phi_yx = np.mean(H_north**3 * d_north(chi1))
    phi_sy = -np.mean(H_north**3 * d_north(psi1))  # h0 b12, (5.325b)

    return phi_x, phi_y, phi_s, phi_xy, phi_yx, phi_sy


def table(roughness, alphas, n=64):
    values = np.array([flow_factors(roughness, alpha, n) for alpha in alphas])

    # The pad solver has a five point stencil and cannot use the cross terms
    cross = np.abs(values[:, 3:]).max()
    if cross > 1e-6 * values[:, :3].min():
        warnings.warn(
            f"cross flow factors up to {cross:.2e} are ignored by the pad solver"
        )

    def interpolant(column):
        spline = CubicSpline(alphas, values[:, column])
        
        
        def phi(H):
            tol = 1e-3 # Accounts for floating point precission 
            if H.min() < alphas[0] - tol or H.max() > alphas[-1] + tol:
                raise ValueError(
                    f"film thickness {H.min():.3f} to {H.max():.3f} is outside "
                    f"the table, alpha {alphas[0]} to {alphas[-1]}"
                )
            return spline(H)

        return phi

    # Passed to cartesian.solve as phi_x, phi_y and phi_s
    return interpolant(0), interpolant(1), interpolant(2)
