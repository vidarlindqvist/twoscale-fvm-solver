"""Load carrying capacity, the pressure integrated over the pad with the trapezoid rule."""

import numpy as np


def cartesian(pressure, X_vector, Y_vector):
    # int P dX dY, multiply by the pressure scale and length**2 for newtons
    return np.trapezoid(np.trapezoid(pressure, X_vector, axis=1), Y_vector)


def polar(pressure, R_vector, theta_vector):
    # int P R dR dtheta, multiply by the pressure scale and Ro**2 for newtons
    R = R_vector[np.newaxis, :]
    return np.trapezoid(np.trapezoid(pressure * R, R_vector, axis=1), theta_vector)
