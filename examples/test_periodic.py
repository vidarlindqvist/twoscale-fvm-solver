import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from reynolds import assembly, boundary, film

nodes = 64
step = 1.0 / nodes

y_1, y_2 = np.meshgrid(np.arange(nodes) * step, np.arange(nodes) * step)


def height(y_1, y_2):
    return 1.0 + 0.4 * np.sin(2 * np.pi * y_1) * np.sin(2 * np.pi * y_2)


east, west, north, south = film.faces(height, y_1, y_2, step, step)

a_east = (east**3 / step**2).flatten()
a_west = (west**3 / step**2).flatten()
a_north = (north**3 / step**2).flatten()
a_south = (south**3 / step**2).flatten()
a_P = a_east + a_west + a_north + a_south

e = [1, 0]
k = 1
source = ( e[0] * step * (east ** k - west ** k) + e[1] * step * (north ** k - south ** k) ).flatten()

pinned = boundary.pin(nodes * nodes)

print("pinned:", pinned)

A = assembly.five_point(
    a_east, a_west, a_north, a_south, a_P, nodes, nodes, pinned, periodic=True
)
rhs = assembly.identity_rhs(source, pinned)

pressure = spsolve(A.tocsr(), rhs).reshape(nodes, nodes)

plt.pcolormesh(y_1, y_2, pressure, cmap="viridis", shading="auto")
plt.colorbar(label="pressure")
plt.xlabel("$y_1$")
plt.ylabel("$y_2$")
plt.gca().set_aspect("equal")
plt.show()
