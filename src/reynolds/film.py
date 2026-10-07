"""Dimensionless film thickness H = h / h_min for the pads and the roughness."""

import numpy as np





def step(H_max, H_min, a, b):
# Step function
    def film_thickness(*geometry):
        def H(X1, X2):
            return np.where((X1 >= a) & (X1 <= b), H_max, H_min) + 0 *X2
        return H
    return film_thickness
        
        
def cartesian(length, width, h_taper, h_min):
    # Real height function
    def h(x1, x2):
        return h_min + h_taper * (length - x1) / length

    # Dimensionless height function, X and Y are both in units of the length
    def H(X1, X2):
        return h(X1 * length, X2 * length) / h_min

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
    # Roughness on the unit cell, with minimum 0 so that alpha is the smallest gap
    def H_r(y1, y2):
        return amplitude * (1 + np.sin(2 * np.pi * y1) * np.sin(2 * np.pi * y2))

    return H_r


def rough(film_thickness, roughness, epsilon):
    # A smooth pad film with the roughness drawn in at wavelength epsilon,
    # used to solve the rough pad directly and compare with the homogenised one
    def rough_film_thickness(*geometry):
        H_0 = film_thickness(*geometry)

        def H(X1, X2):
            return H_0(X1, X2) + roughness(X1 / epsilon, X2 / epsilon)

        return H

    return rough_film_thickness


def faces(H, first, second, d_first, d_second):
    # Film thickness at the east, west, north and south faces
    return (
        H(first + d_first / 2, second),
        H(first - d_first / 2, second),
        H(first, second + d_second / 2),
        H(first, second - d_second / 2),
    )



