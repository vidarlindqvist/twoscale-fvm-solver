"""The homogenised pad against the rough pad with every bump resolved.

As the wavelength epsilon goes to zero the resolved pressure should approach
the homogenised one, p_eps -> p_0. 
"""

import matplotlib.pyplot as plt
import numpy as np

from reynolds import cartesian, film, homogenised, load

roughness = film.bisinusoidal(-0.1)

nodes_per_wavelength = 16   # y1_nodes and y2_nodes
epsilons = [1 / 4, 1 / 8, 1 / 16, 1 / 32]
x1_nodes=256
x2_nodes=256

pad = {"length": 1.0, "width": 1.0, "h_taper": 5e-6, "h_min": 20e-6}

alphas = np.linspace(1, ( pad["h_min"] + pad["h_taper"] ) / pad["h_min"], 32)

# Smooth pad, for reference
P_smooth, X1_vector, X2_vector = cartesian.solve(**pad, x1_nodes=x1_nodes, x2_nodes=x2_nodes)
load_smooth = load.cartesian(P_smooth, X1_vector, X2_vector)
print(f"smooth         load {load_smooth:.6f}")


# Homogenised pad, the flow factor table replaces H**3 and H
phi_x, phi_y, phi_s = homogenised.table(roughness, alphas, n=nodes_per_wavelength)
P_homogenised, X1_vector, X2_vector = cartesian.solve(
    **pad, x1_nodes=x1_nodes, x2_nodes=x2_nodes, phi_x=phi_x, phi_y=phi_y, phi_s=phi_s
)

load_homogenised = load.cartesian(P_homogenised, X1_vector, X2_vector)
load_ratio = load_homogenised / load_smooth
print(f"homogenised    load {load_homogenised:.6f}")
print("-----------------------------")
print(f"              ratio {load_ratio:.6f}")

# roughness() is exactly = 0 in the middle so plot_offset makes sure to offset
# the plot to a peak of roughness to get the most visible difference
plot_offset = nodes_per_wavelength // 4 + 1

fig, ax = plt.subplots(figsize=(8, 5))


# Resolved rough pad, with the same number of nodes per bump each time

for epsilon in epsilons:
    nodes = round(nodes_per_wavelength / epsilon)+1

    P_eps, X1_eps, X2_eps = cartesian.solve(
        **pad,
        x1_nodes=nodes,
        x2_nodes=nodes,
        film_thickness=film.rough(film.cartesian, roughness, epsilon),
    )
    load_eps = load.cartesian(P_eps, X1_eps, X2_eps)
    ax.plot(X1_eps, P_eps[len(X2_eps) // 2 - plot_offset,:],
            "-",
            label=rf"perturbed $\varepsilon$ = 1/{round(1/epsilon)}",
            )

ax.plot(X1_vector, P_smooth[len(X2_vector) // 2 - plot_offset, :],
        "-",
        label="smooth"
        )

ax.plot(X1_vector, P_homogenised[len(X2_vector) // 2 - plot_offset, :],
        label="homogenised",
        color = "black"
        )

ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()
plt.show()
