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


dimless_polar_pressure, R_vector, theta_vector, theta2 = polar.solve(
    Ri=Ri,
    Ro=Ro,
    h0=h0,
    delta_h=delta_h,
    r_nodes=n_span,
    theta_nodes=n_sliding,
    theta1 = theta1,
    theta2 = theta2
)

# LCC is the integal of dimless pressure over dimless area
# The integral equals the mean pressure times the area
LCC = np.mean(dimless_polar_pressure) * (theta2 - theta1) * (Ro ** 2 - Ri ** 2) / 2

# Rescaling lcc
omega = 650e3 / (LCC * 6 * eta0 * Ro ** 2 / h0 ** 2)
lcc = LCC * 6 * eta0 * omega * Ro ** 2 / h0 ** 2

print(LCC)
print(omega)