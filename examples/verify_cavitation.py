"""
Verifies cavitaion.py with a 1D analytical solution for pocket bearing
""" 

import matplotlib.pyplot as plt
import numpy as np

from reynolds import film, cavitation

# Pocket geometry:
a = 0.1
b = 0.25
H_min = 1
H_max = 10
length = 20e-3
width = 2000e-3
p_c = -1e-4 # (p_c - p_in) / (6 mu U l / h0^2)

pad = {"H_max": H_max, "H_min": H_min, "a": a, "b": b}

x1_nodes = 101   # Weird things happens at 11,26,27,36,56,75,91
x2_nodes = 21

X_vector = np.linspace(0, 1, 1001)  # For analytical solution
X1_vector = np.linspace(0, 1, x1_nodes)
X2_vector = np.linspace(0, width / length, x2_nodes)


# Numerical solution
numerical_pressure, cavitation = cavitation.solve(
        length = length,
        width = width,
        x1_nodes = x1_nodes,
        x2_nodes = x2_nodes,
        cavitation_pressure = p_c,
        film_thickness=film.pocket(**pad)
    )


# Analytical 1D solution
def analytic_pocket(X, H_max, H_min, a, b, p_c):
    C1 = p_c / a
    C2 = 0
    C3 = (H_min - H_min ** 3 * C1 - H_max) / ( - H_max ** 3) 
    C5 = C1
    C6 = -C5
    z = (C5 * (b - 1) - C3 * b - p_c) / (- C3)
    C4 = p_c - C3 * z
    
    p = np.select([X <= a, X <= z, X <= b],
                  [C1 * X + C2,
                  p_c * np.ones_like(X),
                  C3 * X + C4],
                  C5 * X + C6
                  )
    return p

analytical_pressure = analytic_pocket(X=X_vector, **pad, p_c=p_c)

# Comparing 

difference = numerical_pressure[x2_nodes // 2, :] - analytic_pocket(X=X1_vector, **pad, p_c=p_c)
rms_error = np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(numerical_pressure**2))
print(f"Relative RMS error   {rms_error:.3e}")

fig, ax = plt.subplots(figsize=(10, 6))
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}


ax.plot(X_vector, analytical_pressure,
         label = "Analytical pressure")
ax.plot(
        X1_vector,numerical_pressure[x2_nodes // 2, :],
        '.', label = "Numerical pressure"
        )

ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()
plt.show()



