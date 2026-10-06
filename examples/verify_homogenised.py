"""The homogenised pad against the rough pad with every bump resolved.

As the wavelength epsilon goes to zero the resolved pressure should approach
the homogenised one, p_eps -> p_0. 
"""

import matplotlib.pyplot as plt
import numpy as np

from reynolds import cartesian, film, homogenised, load

roughness = film.bisinusoidal(-0.025)
alphas = np.linspace(0.9, 1.4, 32)
nodes_per_wavelength = 8
epsilons = [1 / 4, 1 / 8, 1 / 16, 1 / 32]

pad = {"length": 1.0, "width": 1.0, "h_taper": 5e-6, "h_min": 20e-6}

# Smooth pad, for reference
P_smooth, X_vector, Y_vector, _ = cartesian.solve(**pad, x_nodes=321, y_nodes=321)
print(f"smooth         load {load.cartesian(P_smooth, X_vector, Y_vector):.6f}")

# Homogenised pad, the flow factor table replaces H**3 and H
phi_x, phi_y, phi_s = homogenised.table(roughness, alphas, n=nodes_per_wavelength)
P_0, X_vector, Y_vector, _ = cartesian.solve(
    **pad, x_nodes=321, y_nodes=321, phi_x=phi_x, phi_y=phi_y, phi_s=phi_s
)
load_0 = load.cartesian(P_0, X_vector, Y_vector)
print(f"homogenised    load {load_0:.6f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X_vector, P_smooth[160, :], "--", label="smooth")
ax.plot(X_vector, P_0[len(Y_vector) // 2, :])
# Resolved rough pad, with the same number of nodes per bump each time
for epsilon in epsilons:
    nodes = round(nodes_per_wavelength / epsilon)

    P_eps, X_eps, Y_eps, _ = cartesian.solve(
        **pad,
        x_nodes=nodes,
        y_nodes=nodes,
        film_thickness=film.rough(film.cartesian, roughness, epsilon),
    )
    load_eps = load.cartesian(P_eps, X_eps, Y_eps)
    ax.plot(X_eps, P_eps[len(Y_eps) // 2,:], "--", label="perturbed")
plt.show()
