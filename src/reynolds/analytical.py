"""Analytical 1D solution of the Reynolds equation for a linear slider.

Takes the dimensionless sliding coordinate X and the film taper ratio H_taper.
Returns the dimensionless pressure at those positions.
"""


# 1D Analytical solution
def pressure_1D(X, H_taper):
    H = 1 + H_taper * (1 - X)
    return (
        1 / (H_taper * H)
        - (1 + H_taper) / (H_taper * (2 + H_taper) * H**2)
        - 1 / (H_taper * (2 + H_taper))
    )
