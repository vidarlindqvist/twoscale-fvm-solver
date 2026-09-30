"""Analytical 1D pressure for the linear slider, used to verify the cartesian solver."""


def pressure_1D(X, H_taper):
    H = 1 + H_taper * (1 - X)
    return (
        1 / (H_taper * H)
        - (1 + H_taper) / (H_taper * (2 + H_taper) * H**2)
        - 1 / (H_taper * (2 + H_taper))
    )
