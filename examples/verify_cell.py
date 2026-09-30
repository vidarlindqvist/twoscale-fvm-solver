"""The chi1 cell problem on the bi-sinusoidal surface."""

import matplotlib.pyplot as plt
import numpy as np

from reynolds import film, homogenised

# h = 0.1 + 0.5 (1 + sin(2 pi y1) sin(2 pi y2))
alpha = 0.1
roughness = film.bisinusoidal(0.5)
n = 64

solutions, y_vector = homogenised.solve_cell(roughness, alpha, problems=["chi1"], n=n)
chi1 = solutions["chi1"] - solutions["chi1"].mean()
phi_x, phi_y, phi_s, *_ = homogenised.flow_factors(roughness, alpha, n=n)

print(f"chi1 from {chi1.min():.3f} to {chi1.max():.3f}")
print(f"phi_x = {phi_x:.4f}, phi_y = {phi_y:.4f}, phi_s = {phi_s:.4f}")

Y1, Y2 = np.meshgrid(y_vector, y_vector)
fig, ax = plt.subplots(figsize=(6, 5.5))
mesh = ax.pcolormesh(Y1, Y2, chi1, cmap="viridis", shading="auto")
fig.colorbar(mesh, ax=ax, label=r"$\chi_1$")
ax.set_xlabel("$y_1$")
ax.set_ylabel("$y_2$")
ax.set_aspect("equal")

fig.savefig("verify_cell.png", dpi=200, bbox_inches="tight")
plt.show()
