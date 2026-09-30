"""Results of the pad solvers.

The first two tests pin the numbers so refactoring can't change them. The rest
check the solvers against the analytical solution, against each other and
against a separate solver written in metres and pascals.
"""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from reynolds import analytical, assembly, boundary, cartesian, load, polar

CARTESIAN_PEAK = 1.318850627506926e-02
CARTESIAN_SUM = 1.495937284496961e01
# Re-recorded after the polar source term gained its missing factor R.
POLAR_PEAK = 1.4963461910380317e-07
POLAR_SUM = 1.4678174666525625e-04


def test_cartesian_default_result_is_unchanged():
    pressure = cartesian.solve()[0]

    assert pressure.max() == pytest.approx(CARTESIAN_PEAK, rel=1e-10)
    assert pressure.sum() == pytest.approx(CARTESIAN_SUM, rel=1e-10)


def test_polar_default_result_is_unchanged():
    pressure = polar.solve()[0]

    assert pressure.max() == pytest.approx(POLAR_PEAK, rel=1e-10)
    assert pressure.sum() == pytest.approx(POLAR_SUM, rel=1e-10)


def test_cartesian_matches_the_analytical_solution_on_a_wide_pad():
    # Little side leakage on a wide pad, so mid span should match 1D
    pressure, X_vector, _, H_taper = cartesian.solve(
        length=1.0, width=10.0, x_nodes=80, y_nodes=80
    )

    mid_span = pressure[80 // 2, :]
    exact = analytical.pressure_1D(X_vector, H_taper)
    error = np.sqrt(np.mean((mid_span - exact) ** 2)) / np.sqrt(np.mean(exact**2))

    assert error < 1e-4


def test_polar_matches_cartesian_on_a_matched_square_pad():
    # A thin annulus with arc length = radial width is almost a square pad
    Ri, Ro = 299e-3, 300e-3
    theta2 = (Ro - Ri) / Ri

    polar_pressure = polar.solve(
        Ri=Ri, Ro=Ro, r_nodes=50, theta_nodes=50, theta1=0, theta2=theta2
    )[0]
    cartesian_pressure = cartesian.solve(
        length=1.0, width=1.0, x_nodes=50, y_nodes=50
    )[0]

    # polar scales pressure with omega * Ro**2, cartesian with u_s * length.
    # Here length = Ro - Ri and the pad slides at u_s = omega * r, about omega
    # times the mean radius, so P_cartesian = P_polar * Ro**2 / (R_mean * (Ro - Ri)).
    R_mean = (Ri + Ro) / 2
    rescaled = polar_pressure.T * Ro**2 / (R_mean * (Ro - Ri))
    error = np.sqrt(np.mean((cartesian_pressure - rescaled) ** 2)) / np.sqrt(
        np.mean(cartesian_pressure**2)
    )

    assert error < 1e-3


def test_polar_matches_an_independent_dimensional_solver_on_a_wide_annulus():
    # R is close to 1 on the thin annulus, so the real bearing is used here
    Ri, Ro, h_min, h_taper = 250e-3, 500e-3, 20e-6, 5e-6
    theta2, mu, omega, n = 55 * np.pi / 180, 30e-3, 1.0, 60
    slope = h_taper / ((Ri + Ro) / 2 * np.sin(theta2))

    def h(r, theta):
        return h_min + slope * r * np.sin(theta2 - theta)

    r_vector = np.linspace(Ri, Ro, n)
    theta_vector = np.linspace(0, theta2, n)
    r, theta = np.meshgrid(r_vector, theta_vector)
    dr, dtheta = r_vector[1] - r_vector[0], theta_vector[1] - theta_vector[0]

    # d/dr(r h^3/12mu dp/dr) + 1/r d/dtheta(h^3/12mu dp/dtheta) = r omega/2 dh/dtheta
    a_east = ((r + dr / 2) * h(r + dr / 2, theta) ** 3 / (12 * mu) * dtheta / dr).flatten()
    a_west = ((r - dr / 2) * h(r - dr / 2, theta) ** 3 / (12 * mu) * dtheta / dr).flatten()
    a_north = (dr / (r * dtheta) * h(r, theta + dtheta / 2) ** 3 / (12 * mu)).flatten()
    a_south = (dr / (r * dtheta) * h(r, theta - dtheta / 2) ** 3 / (12 * mu)).flatten()
    a_P = a_east + a_west + a_north + a_south
    d_sin = np.sin(theta2 - theta - dtheta / 2) - np.sin(theta2 - theta + dtheta / 2)
    source = -(omega / 2 * slope * d_sin * ((r + dr / 2) ** 3 - (r - dr / 2) ** 3) / 3).flatten()

    fixed = boundary.all_edges(n, n)
    A = assembly.five_point(a_east, a_west, a_north, a_south, a_P, n, n, fixed)
    p = spsolve(A.tocsr(), assembly.identity_rhs(source, fixed)).reshape(n, n)
    force = load.polar(p, r_vector, theta_vector)

    P, R_vector, theta_vector, _ = polar.solve(
        Ri=Ri, Ro=Ro, h_min=h_min, h_taper=h_taper, r_nodes=n, theta_nodes=n, theta2=theta2
    )
    pressure_scale = 6 * mu * omega * Ro**2 / h_min**2
    scaled_force = pressure_scale * Ro**2 * load.polar(P, R_vector, theta_vector)

    assert scaled_force == pytest.approx(force, rel=1e-10)
