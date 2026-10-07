import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import spsolve
from reynolds import cartesian, film, assembly, boundary

length = 50e-3
width = 50e-3
H_min = 1
H_max = 10
x1_nodes = 5
x2_nodes = 5
a=0.1
b=0.2
h_taper=5e-6
h_min = 20e-6

dX2 = (width / length) / (x2_nodes - 1)
dX1 = 1 / (x1_nodes - 1)
cavitation_pressure, X1_vector, X2_vector = cartesian.solve(
    length=length,
    width=width,
    h_taper=h_taper,
    h_min=h_min,
    x1_nodes=x1_nodes,
    x2_nodes=x2_nodes,
    film_thickness = film.step(H_min,H_max,a,b)
)

Xs = np.linspace(0, 1, x1_nodes)

print(f"Cartesian aspect ratio   {length / width:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}

ax.plot(X1_vector, cavitation_pressure[x2_nodes // 2, :], label="Cartesian FVM")


ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()

fig.savefig("verify_cartesian.png", dpi=200, bbox_inches="tight")
plt.show()


H_east, H_west, H_north, H_south = film.faces(
    film.step(H_min,H_max,a,b),
    X1_vector,
    X2_vector,
    dX1,
    dX2
    )

fixed = boundary.all_edges(x1_nodes, x2_nodes)

H_test = H_east ** 3
a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
    H_test,
    H_west ** 3,
    H_north ** 3,
    H_south ** 3,
    dX1,
    dX2
    )

A = assembly.five_point(
    a_east,
    a_west,
    a_north,
    a_south,
    a_P,
    x1_nodes * x2_nodes,
    x1_nodes * x2_nodes,
    fixed,
    periodic=False,
    )


b_east, b_west, b_P = assembly.cavitation_coefficiens(
    H_east,
    H_west,
    dX2
    )
b_north = np.zeros_like(b_west)
b_south = np.zeros_like(b_west)




B = assembly.five_point(
    b_east,
    b_west,
    b_north,
    b_south,
    b_P,
    x1_nodes * x2_nodes,
    x1_nodes * x2_nodes,
    fixed,
    periodic=False,
    )

source = assembly.build_source(
    H_east,
    H_west,
    H_north,
    H_south,
    dX1,
    dX2,
    (1,0)
    )
    
rhs = assembly.identity_rhs(source, fixed)

pressure = spsolve(A.tocsr(), rhs).reshape(x2_nodes, x1_nodes)

compare = np.zeros_like(pressure) < 0





