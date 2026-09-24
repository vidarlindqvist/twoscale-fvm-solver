"""Load carrying capacity of the hydropower thrust bearing from Table 1.

Takes the bearing geometry, the lubricant viscosity and the target load.
Prints the dimensionless LCC and the shaft speed needed to reach that load.
"""

import numpy as np

from reynolds import load, polar

Ri = 250e-3
Ro = 500e-3
h_min = 20e-6
h_taper = 5e-6
n_sliding = 50
n_span = 50
theta1 = 0
theta2 = 55 * np.pi / 180
eta0 = 30e-3
target_load = 650e3 # Newton


dimless_polar_pressure, R_vector, theta_vector, _ = polar.solve(
    Ri=Ri,
    Ro=Ro,
    h_min=h_min,
    h_taper=h_taper,
    r_nodes=n_span,
    theta_nodes=n_sliding,
    theta1=theta1,
    theta2=theta2,
)

# LCC (dimless lcc) is the integal of dimless pressure over dimless area,
# int P R dR dtheta. The area element carries the factor R.
LCC = load.polar(dimless_polar_pressure, R_vector, theta_vector)

# Back to newtons: p = P * 6 * eta0 * omega * Ro**2 / h_min**2 and dA = Ro**2 R dR dtheta,
# so load = LCC * 6 * eta0 * omega * Ro**4 / h_min**2. Solve that for omega.
omega = target_load / (LCC * 6 * eta0 * Ro**4 / h_min**2)

print("LCC:", LCC)
print("omega:", omega, "rad/s =", omega * 60 / (2 * np.pi), "rpm")
