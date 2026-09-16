"""Analytical 1D solution of the Reynolds equation for a linear slider.

Takes the dimensionless sliding coordinate Xs and the film taper ratio delta_H.
Returns the dimensionless pressure at those positions.
"""


# 1D Analytical solution
def pressure_1D(Xs, delta_H):
    H = 1 + delta_H * (1 - Xs)
    return (
        1 / (delta_H * H)
        - (1 + delta_H) / (delta_H * (2 + delta_H) * H**2)
        - 1 / (delta_H * (2 + delta_H))
    )
