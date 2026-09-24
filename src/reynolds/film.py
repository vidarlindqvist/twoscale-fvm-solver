import numpy as np


def cartesian(length, width, h_taper, h_min):
    # Real height function
    def h(x, y):
        return h_min + h_taper * (length - x) / length

    # Dimensionless height function. Both X and Y are in units of the length.
    def H(X, Y):
        return h(X * length, Y * length) / h_min

    return H


def polar(Ri, Ro, h_taper, h_min, theta2):
    # Real height function
    def h(r, theta):
        return h_min + h_taper * r * np.sin(theta2 - theta) / (
            (Ri + Ro) / 2 * np.sin(theta2)
        )

    # Dimensionless height function
    def H(R, theta):
        return h(R * Ro, theta) / h_min

    return H


def bisinusoidal(amplitude):
    def H_r(y1, y2):
        return amplitude * (1 + np.sin(2 * np.pi * y1) * np.sin(2 * np.pi * y2))

    return H_r


def faces(H, first, second, d_first, d_second):
    return (
        H(first + d_first / 2, second),
        H(first - d_first / 2, second),
        H(first, second + d_second / 2),
        H(first, second - d_second / 2),
    )
