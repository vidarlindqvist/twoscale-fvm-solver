import matplotlib.pyplot as plt
import numpy as np

from reynolds import cell, film

alpha = 0.1
roughness = film.bisinusoidal(0.5)
nodes = 64


chi1, y_vector = cell.solve(roughness, alpha, "chi1", n=nodes)
phi_x, phi_y, phi_s, phi_xy, phi_yx, phi_sy = cell.flow_factors(roughness, alpha, n=nodes)

Y1, Y2 = np.meshgrid(y_vector, y_vector)

fig, ax = plt.subplots(figsize=(6, 5.5))
mesh = ax.pcolormesh(Y1, Y2, chi1 - chi1.mean(), cmap="viridis", shading="auto")
fig.colorbar(mesh, ax=ax)
ax.set_xlabel("$y_1$")
ax.set_ylabel("$y_2$")
ax.set_aspect("equal")

fig.savefig("verify_cell.png", dpi=200, bbox_inches="tight")
plt.show()
