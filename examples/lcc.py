"""Load carrying capacity of the hydropower thrust bearing from Table 1.

Takes the bearing geometry, the lubricant viscosity and the target load.
Prints the dimensionless LCC and the shaft speed needed to reach that load.
"""

import numpy as np

from reynolds import polar

Ri = 250e-3
Ro = 500e-3
h0 = 20e-6
delta_h = 5e-6
n_sliding = 50
n_span = 50
theta1 = 0
theta2 = 55 * np.pi / 180
eta0 = 30e-3
lcc = 650e3 # Newton


dimless_polar_pressure, R_vector, theta_vector, _ = polar.solve(
    Ri=Ri,
    Ro=Ro,
    h0=h0,
    delta_h=delta_h,
    r_nodes=n_span,
    theta_nodes=n_sliding,
    theta1=theta1,
    theta2=theta2,
)

# LCC (dimless lcc) is the integal of dimless pressure over dimless area
LCC = 6 * np.mean(dimless_polar_pressure) * (theta2 - theta1) * (Ro**2 - Ri**2) / 2

# Evaluating the required angular velocity
omega = lcc / (LCC * 6 * eta0 * Ro**2 / h0**2)

print("LCC:", LCC)
print("omega:", omega)
