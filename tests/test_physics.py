"""Physics invariants that must hold regardless of geometry or discretisation.

Takes no inputs of its own, each test drives the solvers or the film definitions
directly.
Fails if the wedge, the boundary condition or the film taper stops behaving.
"""

import numpy as np
import pytest

from reynolds import cartesian, polar

SOLVERS = {
    "cartesian": cartesian.solve,
    "polar": polar.solve,
}


def flat_film(*geometry):
    """A film of uniform thickness, ignoring whatever geometry it is given."""

    def H(first, second):
        return np.ones_like(first * second)

    return H


@pytest.mark.parametrize("name", SOLVERS)
def test_flat_film_generates_no_pressure(name):
    pressure = SOLVERS[name](film_thickness=flat_film)[0]

    assert np.all(pressure == 0.0)


@pytest.mark.parametrize("name", SOLVERS)
def test_boundary_pressure_is_exactly_zero(name):
    pressure = SOLVERS[name]()[0]

    assert np.all(pressure[0, :] == 0.0)
    assert np.all(pressure[-1, :] == 0.0)
    assert np.all(pressure[:, 0] == 0.0)
    assert np.all(pressure[:, -1] == 0.0)


@pytest.mark.parametrize("name", SOLVERS)
def test_converging_wedge_generates_positive_pressure(name):
    pressure = SOLVERS[name]()[0]

    assert np.all(pressure[1:-1, 1:-1] > 0.0)
