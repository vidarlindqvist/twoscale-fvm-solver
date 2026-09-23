"""Solution of the periodic FVM problem.

Solves on a square pad and shows the pressure field as an interactive 3D
surface that can be rotated and zoomed.
"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from reynolds import periodic

l = 500e-3
b = 500e-3
delta_h = 5e-6
h0 = 20e-6
x_nodes = 100
y_nodes = 100
k = 1
e = [1, 0]


periodic_pressure, X_vector, Y_vector, _ = periodic.solve(
    l=l,
    b=b,
    delta_h=delta_h,
    h0=h0,
    x_nodes=x_nodes,
    y_nodes=y_nodes,
    k=k,
    e=e,
)

X, Y = np.meshgrid(X_vector, Y_vector)

fig = plt.figure(figsize=(10, 7))
ax = cast(Axes3D, fig.add_subplot(projection="3d"))
surf = ax.plot_surface(X, Y, periodic_pressure, cmap="viridis", linewidth=0)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Dimensionless pressure")

ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless span direction")
ax.set_zlabel("Dimensionless pressure", labelpad=12)

plt.show()
