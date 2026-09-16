"""Verification of the polar FVM solution against the cartesian one.

Solves both on matched near-square domains and compares them node by node.
Prints the pad aspect ratio and the relative RMS error, and saves verify_polar.png.
"""

import matplotlib.pyplot as plt
import numpy as np

from reynolds import cartesian, polar

Ri = 299e-3
Ro = 300e-3
h0 = 20e-6
delta_h = 5e-6
n_sliding = 50
n_span = 50

cartesian_pressure, X_vector, Y_vector, delta_H = cartesian.solve(
    l=1.0,
    b=1.0,
    delta_h=delta_h,
    h0=h0,
    x_nodes=n_sliding,
    y_nodes=n_span,
)

polar_pressure, R_vector, theta_vector, theta2 = polar.solve(
    Ri=Ri,
    Ro=Ro,
    h0=h0,
    delta_h=delta_h,
    r_nodes=n_span,
    theta_nodes=n_sliding,
    theta1=0,
    theta2=(Ro - Ri) / Ri,
)

polar_scale = Ro / (Ro - Ri)
polar_matched = polar_pressure.T * polar_scale

difference = cartesian_pressure - polar_matched
rms_error = np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(cartesian_pressure**2))
aspect_ratio = (Ri + Ro) / 2 * theta2 / (Ro - Ri)

print(f"polar aspect ratio   {aspect_ratio:.4f}")
print(f"relative RMS error   {rms_error:.3e}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_vector, cartesian_pressure[n_span // 2, :], label="Cartesian FVM")
ax.plot(
    theta_vector / theta2,
    polar_matched[n_span // 2, :],
    "--",
    label="Polar FVM (rescaled)",
)

ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()

text_str = f"Aspect ratio: {aspect_ratio: .2e}\nRMS Error: {rms_error: .2e}"
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}

ax.text(
    0.775, 0.85, text_str, transform=ax.transAxes, verticalalignment="top", bbox=bbox
)
fig.savefig("verify_polar.png", dpi=200, bbox_inches="tight")
plt.show()
