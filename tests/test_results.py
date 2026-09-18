"""Checks that the solvers keep giving the answers they give today.

Takes no inputs of its own, each test calls a solver directly.
Fails if a refactor changes a result, which is the point: these are here so that
moving code around can be done without wondering whether it broke the physics.

The first two tests pin numbers. The last two say why those numbers are right,
by checking the solvers against the analytical solution and against each other.
"""

import numpy as np
import pytest

from reynolds import analytical, cartesian, polar

# Recorded from the solvers as they are now. A refactor should not move these.
CARTESIAN_PEAK = 1.318850627506926e-02
CARTESIAN_SUM = 1.495937284496961e01
POLAR_PEAK = 1.497854119832232e-07
POLAR_SUM = 1.469286712470839e-04


def test_cartesian_default_result_is_unchanged():
    pressure = cartesian.solve()[0]

    assert pressure.max() == pytest.approx(CARTESIAN_PEAK, rel=1e-10)
    assert pressure.sum() == pytest.approx(CARTESIAN_SUM, rel=1e-10)


def test_polar_default_result_is_unchanged():
    pressure = polar.solve()[0]

    assert pressure.max() == pytest.approx(POLAR_PEAK, rel=1e-10)
    assert pressure.sum() == pytest.approx(POLAR_SUM, rel=1e-10)


def test_cartesian_matches_the_analytical_solution_on_a_wide_pad():
    """A wide pad leaks little out of the sides, so the mid span slice should
    sit on top of the 1D analytical solution."""
    pressure, X_vector, _, delta_H = cartesian.solve(
        l=1.0, b=10.0, x_nodes=80, y_nodes=80
    )

    mid_span = pressure[80 // 2, :]
    exact = analytical.pressure_1D(X_vector, delta_H)
    error = np.sqrt(np.mean((mid_span - exact) ** 2)) / np.sqrt(np.mean(exact**2))

    assert error < 1e-4


def test_polar_matches_cartesian_on_a_matched_square_pad():
    """A thin annulus is almost a rectangle. Choosing theta2 so the arc length at
    the inner radius equals the radial width makes the pad square, and then the
    two solvers must agree once the different reference lengths are undone."""
    Ri, Ro = 299e-3, 300e-3
    theta2 = (Ro - Ri) / Ri

    polar_pressure = polar.solve(
        Ri=Ri, Ro=Ro, r_nodes=50, theta_nodes=50, theta1=0, theta2=theta2
    )[0]
    cartesian_pressure = cartesian.solve(l=1.0, b=1.0, x_nodes=50, y_nodes=50)[0]

    # polar normalises radius by Ro, cartesian normalises by the pad length
    rescaled = polar_pressure.T * (Ro / (Ro - Ri))
    error = np.sqrt(np.mean((cartesian_pressure - rescaled) ** 2)) / np.sqrt(
        np.mean(cartesian_pressure**2)
    )

    assert error < 1e-3
