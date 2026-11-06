"""Random roughness for the cell problems.

The film is H = alpha + H_r with mean(H_r) = 0, so alpha is the mean gap.
Sq / alpha has to stay below about 0.2 or the film goes negative.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .fractal_surface import fractal_surface


def generate(seed, hurst=0.5, anisotropy=1.0, nodes=256, **options):
    # z[i, j] is the height at y2 = i / nodes, y1 = j / nodes
    # isotropicBand keeps the flow factors from jumping with anisotropy
    options.setdefault("isotropicBand", True)
    surface = fractal_surface(
        {"Nx": nodes, "Ny": nodes, "plot": False, "outputPrefix": ""},
        seed=int(seed), H=hurst, anisotropy=anisotropy, **options,
    )
    return surface.z


def periodic(z, sq):
    # Roughness function H_r(y1, y2), bilinear between the nodes
    heights = (z - z.mean()) / z.std() * sq

    # Repeat the first row and column so the last node connects to the first
    heights = np.pad(heights, ((0, 1), (0, 1)), mode="wrap")
    y2 = np.linspace(0, 1, heights.shape[0])
    y1 = np.linspace(0, 1, heights.shape[1])
    interpolate = RegularGridInterpolator((y2, y1), heights)

    def H_r(y1, y2):
        y1, y2 = np.broadcast_arrays(np.mod(y1, 1), np.mod(y2, 1))
        return interpolate(np.stack([y2, y1], axis=-1))

    return H_r
