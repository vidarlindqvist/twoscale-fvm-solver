"""
Testing cavitation and homogenisation at the same time
"""
import matplotlib.pyplot as plt
import numpy as np

from reynolds import cavitation, film, homogenised

# Geometry
a = 0.1
b = 0.25
H_min = 1
H_max = 10
length = 1000e-3
width = 1000e-3
p_c = -1e-4
amplitude = 0.01


film_thickness = film.pocket(H_max, H_min, a, b)
# film_thickness = film.tilted_pad(length, width, 5e-6, 20e-6)

x1_nodes = 81
x2_nodes = 21

X1_vector = np.linspace(0, 1, x1_nodes)
X2_vector = np.linspace(0, width / length, x2_nodes)

inputs = {"length": length, "width": width, "x1_nodes":x1_nodes,
          "x2_nodes":x2_nodes, "cavitation_pressure": p_c,
          "film_thickness": film_thickness}


roughness = film.bisinusoidal(-amplitude)
epsilons = [1 / 16, 1 / 32]

nodes_per_wavelength = 9  # y1_nodes and y2_nodes
alphas = np.linspace(H_min - amplitude, H_max + amplitude, 100)

# Calling flow factors
phi_x, phi_y, phi_s = homogenised.table(roughness,
                                        alphas,
                                        n=nodes_per_wavelength*2)


# Smooth solution for reference
P_smooth, smooth_cavitation = cavitation.solve(**inputs)


# Homogenised solution 

P_homogenised, homogenised_cavitation = cavitation.solve(
        **inputs, 
        phi_x=phi_x,
        phi_y=phi_y,
        phi_s=phi_s,
        )

# Fully resolved solutions
plot_offset = nodes_per_wavelength // 4 + 1
fig, ax = plt.subplots(figsize=(8, 5))

for epsilon in epsilons:
    nodes = round(nodes_per_wavelength / epsilon)+1
    X1_eps = np.linspace(0, 1, nodes)
    X2_eps = np.linspace(0, width / length, nodes)
    inputs["x1_nodes"] = nodes
    inputs["x2_nodes"] = nodes
    inputs["film_thickness"] = film.rough(film_thickness,
                               roughness,
                               epsilon)
    P_eps, cav_eps = cavitation.solve(**inputs)
   
    ax.plot(X1_eps, P_eps[len(X2_eps) // 2 - plot_offset,:],
                "-",
                label=rf"perturbed $\varepsilon$ = 1/{round(1/epsilon)}",
                )



ax.plot(X1_vector, P_smooth[len(X2_vector) // 2, :],
        "-",
        label="smooth"
        )

ax.plot(X1_vector, P_homogenised[len(X2_vector) // 2, :],
        label="homogenised",
        color = "black"
        )

ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()
plt.show()