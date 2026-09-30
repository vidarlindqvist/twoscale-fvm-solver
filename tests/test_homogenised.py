"""Tests of the cell problems, the flow factor table and the homogenised pad.

The exact values come from ridges that only vary along y1, where the cell
problems reduce to ODEs. The flux H**3 (1 + chi1') is then constant, and
periodicity gives phi_x = 1 / mean(H**-3). Along the ridges phi_y = mean(H**3),
and from psi1, phi_s = mean(H**-2) / mean(H**-3).
"""

import numpy as np
import pytest

from reynolds import cartesian, film, homogenised, load

N = 64
ALPHAS = np.linspace(0.9, 1.4, 26)


def flat(y1, y2):
    return 0 * y1


def ridges(y1, y2):
    # With alpha = 0.5 the film is H = 1 + 0.5 sin(2 pi y1)
    return 0.5 + 0.5 * np.sin(2 * np.pi * y1) + 0 * y2


def test_flat_cell_gives_the_smooth_flow_factors():
    alpha = 1.3
    solutions, _ = homogenised.solve_cell(flat, alpha, problems=homogenised.CELL_PROBLEMS, n=N)
    phi_x, phi_y, phi_s, *_ = homogenised.flow_factors(flat, alpha, n=N)

    for u in solutions.values():
        assert np.all(u == 0.0)
    assert phi_x == pytest.approx(alpha**3, rel=1e-14)
    assert phi_y == pytest.approx(alpha**3, rel=1e-14)
    assert phi_s == pytest.approx(alpha, rel=1e-14)


def test_ridges_give_the_exact_flow_factors():
    phi_x, phi_y, phi_s, *_ = homogenised.flow_factors(ridges, alpha=0.5, n=N)

    assert phi_x == pytest.approx(np.sqrt(3) / 4, rel=1e-12)
    assert phi_y == pytest.approx(11 / 8, rel=1e-12)
    assert phi_s == pytest.approx(2 / 3, rel=1e-12)


def test_bisinusoidal_roughness_has_no_cross_terms():
    *_, phi_xy, phi_yx, phi_sy = homogenised.flow_factors(film.bisinusoidal(0.3), 1.0, n=N)

    assert max(abs(phi_xy), abs(phi_yx), abs(phi_sy)) < 1e-12


def test_oblique_ridges_warn_about_cross_terms():
    def oblique(y1, y2):
        return 0.5 + 0.5 * np.sin(2 * np.pi * (y1 + y2))

    with pytest.warns(UserWarning, match="cross flow factors"):
        homogenised.table(oblique, ALPHAS, n=16)


def test_flat_table_reproduces_the_smooth_solver():
    # alpha**3 and alpha are cubics, so the spline represents them exactly
    phi_x, phi_y, phi_s = homogenised.table(flat, ALPHAS, n=8)
    pressure = cartesian.solve(phi_x=phi_x, phi_y=phi_y, phi_s=phi_s)[0]

    assert np.max(np.abs(pressure - cartesian.solve()[0])) < 1e-12


def test_film_outside_the_table_is_refused():
    phi_x, phi_y, phi_s = homogenised.table(flat, np.linspace(1.1, 1.4, 10), n=8)

    with pytest.raises(ValueError, match="outside the table"):
        cartesian.solve(phi_x=phi_x, phi_y=phi_y, phi_s=phi_s)


def test_resolved_rough_pad_converges_to_the_homogenised_pad():
    roughness = film.bisinusoidal(0.2)
    pad = {"length": 1.0, "width": 1.0}

    phi_x, phi_y, phi_s = homogenised.table(roughness, ALPHAS, n=20)
    P_0, X_vector, Y_vector, _ = cartesian.solve(
        **pad, x_nodes=161, y_nodes=161, phi_x=phi_x, phi_y=phi_y, phi_s=phi_s
    )
    load_0 = load.cartesian(P_0, X_vector, Y_vector)

    # 20 nodes per bump on the rough pad, as in the cell
    errors = []
    for epsilon in [1 / 8, 1 / 16]:
        nodes = round(20 / epsilon) + 1
        P_eps, X_eps, Y_eps, _ = cartesian.solve(
            **pad,
            x_nodes=nodes,
            y_nodes=nodes,
            film_thickness=film.rough(film.cartesian, roughness, epsilon),
        )
        errors.append(abs(load.cartesian(P_eps, X_eps, Y_eps) / load_0 - 1))

    # First order in epsilon, the error halves when epsilon does
    assert errors[1] < 0.005
    assert errors[0] / errors[1] == pytest.approx(2, abs=0.1)
