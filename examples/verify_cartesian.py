"""Verification of the cartesian FVM solution against the analytical one.

Solves a high aspect ratio pad where side leakage is negligible.
Prints the aspect ratio and the relative RMS error, and saves verify_cartesian.png.
"""

import matplotlib.pyplot as plt
import numpy as np

from reynolds import analytical, cartesian

length = 148e-3
width = 500e-3
h_taper = 5e-6
h_min = 20e-6
x_nodes = 5
y_nodes = 5


cartesian_pressure, X_vector, Y_vector, H_taper = cartesian.solve(
    length=length,
    width=width,
    h_taper=h_taper,
    h_min=h_min,
    x_nodes=x_nodes,
    y_nodes=y_nodes,
)


Xs = np.linspace(0, 1, x_nodes)
analytical_pressure = analytical.pressure_1D(Xs, H_taper)

difference = cartesian_pressure[y_nodes // 2, :] - analytical_pressure
rms_error = np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(cartesian_pressure**2))

print(f"Cartesian aspect ratio   {length / width:.4f}")
print(f"Relative RMS error   {rms_error:.3e}")

fig, ax = plt.subplots(figsize=(10, 6))
text_str = f"Aspect ratio: {length / width}\nRMS Error: {rms_error: .2e}"
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}

ax.text(
    0.80, 0.85, text_str, transform=ax.transAxes, verticalalignment="top", bbox=bbox
)
ax.plot(X_vector, cartesian_pressure[y_nodes // 2, :], label="Cartesian FVM")
ax.plot(Xs, analytical_pressure, "--", label="Analytical 1D")


ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()

fig.savefig("verify_cartesian.png", dpi=200, bbox_inches="tight")
plt.show()
