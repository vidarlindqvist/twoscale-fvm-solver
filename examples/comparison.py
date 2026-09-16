import matplotlib.pyplot as plt
import numpy as np

from reynolds import analytical, cartesian, polar

l = 500e-3
b = 500e-3
delta_h = 5e-6
h0 = 20e-6
x_nodes = 50
y_nodes = 50

Ri = 299e-3
Ro = 300e-3
r_nodes = 50
theta_nodes = 50

cartesian_pressure, X_vector, Y_vector, delta_H = cartesian.solve(
    l=l,
    b=b,
    delta_h=delta_h,
    h0=h0,
    x_nodes=x_nodes,
    y_nodes=y_nodes,
)

polar_pressure, R_vector, theta_vector, theta2 = polar.solve(
    Ri=Ri,
    Ro=Ro,
    h0=h0,
    delta_h=delta_h,
    r_nodes=r_nodes,
    theta_nodes=theta_nodes,
)

Xs = np.linspace(0, 1, x_nodes)
analytical_pressure = analytical.pressure_1D(Xs, delta_H)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Xs, analytical_pressure, label="Analytical 1D")
ax.plot(X_vector, cartesian_pressure[y_nodes // 2, :], label="Cartesian FVM")
ax.plot(theta_vector / theta2, polar_pressure[:, r_nodes // 2], label="Polar FVM")

ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()

fig.savefig("comparison.png", dpi=200, bbox_inches="tight")
plt.show()
