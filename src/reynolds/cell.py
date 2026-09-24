import numpy as np
from scipy.sparse.linalg import splu

from . import assembly, boundary, film

# Power of H on the right hand side, direction, sign
PROBLEMS = {
    "psi1": (1, (1, 0), -1),
    "psi2": (1, (0, 1), -1),
    "chi1": (3, (1, 0), 1),
    "chi2": (3, (0, 1), 1),
}


def local_film(roughness, alpha, n):
    y_vector = np.arange(n) / n
    dy = 1 / n
    Y1, Y2 = np.meshgrid(y_vector, y_vector)

    def H(y1, y2):
        return alpha + roughness(y1, y2)

    H_east, H_west, H_north, H_south = film.faces(H, Y1, Y2, dy, dy)

    return y_vector, dy, H_east, H_west, H_north, H_south


def factorise(faces, dy):
    # The operator is the same for every cell problem, so it is factorised once
    H_east, H_west, H_north, H_south = faces
    n = H_east.shape[0]

    a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
        H_east**3, H_west**3, H_north**3, H_south**3, dy, dy
    )

    fixed = boundary.pin(n * n)
    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, n, n, fixed, periodic=True
    )

    return splu(A.tocsc())


def solve_with(lu, faces, dy, problem):
    H_east, H_west, H_north, H_south = faces
    n = H_east.shape[0]
    k, e, sign = PROBLEMS[problem]

    source = sign * assembly.build_source(
        H_east**k, H_west**k, H_north**k, H_south**k, dy, dy, e
    )
    rhs = assembly.identity_rhs(source, boundary.pin(n * n))

    return lu.solve(np.asarray(rhs, dtype=float)).reshape(n, n)


def solve(roughness, alpha, problem, n=64):
    y_vector, dy, *faces = local_film(roughness, alpha, n)
    lu = factorise(faces, dy)

    return solve_with(lu, faces, dy, problem), y_vector


def flow_factors(roughness, alpha, n=64):
    _, dy, *faces = local_film(roughness, alpha, n)
    H_east, _, H_north, _ = faces
    lu = factorise(faces, dy)

    chi1 = solve_with(lu, faces, dy, "chi1")
    chi2 = solve_with(lu, faces, dy, "chi2")
    psi1 = solve_with(lu, faces, dy, "psi1")

    # Differences across the east and north faces, wrapping around the cell
    dchi1_y1 = (np.roll(chi1, -1, axis=1) - np.roll(chi1, 1, axis=1)) / (2 * dy)
    dchi1_y2 = (np.roll(chi1, -1, axis=0) - np.roll(chi1, 1, axis=0)) / (2 * dy)
    dchi2_y1 = (np.roll(chi2, -1, axis=1) - np.roll(chi2, 1, axis=1)) / (2 * dy)
    dchi2_y2 = (np.roll(chi2, -1, axis=0) - np.roll(chi2, 1, axis=0)) / (2 * dy)
    dpsi1_y1 = (np.roll(psi1, -1, axis=1) - np.roll(psi1, 1, axis=1)) / (2 * dy)
    dpsi1_y2 = (np.roll(psi1, -1, axis=0) - np.roll(psi1, 1, axis=0)) / (2 * dy)

    # Cell averages of the face fluxes
    phi_x = np.mean(H_east**3 * (1 + dchi1_y1))
    phi_y = np.mean(H_north**3 * (1 + dchi2_y2))
    phi_s = np.mean(H_east - H_east**3 * dpsi1_y1)

    # Cross terms
    phi_xy = np.mean(H_east**3 * dchi2_y1)
    phi_yx = np.mean(H_north**3 * dchi1_y2)
    phi_sy = -np.mean(H_north**3 * dpsi1_y2)

    return phi_x, phi_y, phi_s, phi_xy, phi_yx, phi_sy
