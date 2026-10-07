import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags_array

from scipy.sparse.linalg import spsolve
from reynolds import cartesian, film, assembly, boundary

length = 50e-3
width = 50e-3
H_min = 1
H_max = 10
x1_nodes = 11
x2_nodes = 11
a=0.1
b=0.2
h_taper=5e-6
h_min = 20e-6

dX2 = (width / length) / (x2_nodes - 1)
dX1 = 1 / (x1_nodes - 1)

X1_vector = np.linspace(0, 1, x1_nodes)
X2_vector = np.linspace(0, width / length, x2_nodes)

X1, X2 = np.meshgrid(X1_vector, X2_vector)
H = film.step(H_min, H_max, a, b)(length, width, h_taper, h_min)

H_east, H_west, H_north, H_south = film.faces(H, X1, X2, dX1, dX2)


cavitation_pressure, X1_vector, X2_vector = cartesian.solve(
    length=length,
    width=width,
    h_taper=h_taper,
    h_min=h_min,
    x1_nodes=x1_nodes,
    x2_nodes=x2_nodes,
    film_thickness = film.step(H_min,H_max,a,b)
)


fixed = boundary.all_edges(x1_nodes, x2_nodes)


a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
    H_east ** 3,
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
    x1_nodes,
    x2_nodes,
    fixed,
    periodic=False,
    )


# b_east, b_west, b_P = assembly.cavitation_coefficients(
#     H_east,
#     H_west,
#     dX2
#     )


b_P = H_east.flatten()
b_west = H_west.flatten() * dX2
b_east = np.zeros_like(b_west) * dX2
b_north = np.zeros_like(b_west)
b_south = np.zeros_like(b_west)


B = assembly.five_point(
    b_east,
    b_west,
    b_north,
    b_south,
    b_P,
    x1_nodes,
    x2_nodes,
    fixed,
    False,
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
    
rhs = -assembly.identity_rhs(source, fixed)

pressure = spsolve(A.tocsr(), rhs).reshape(x2_nodes, x1_nodes)
cavitation = spsolve(B.tocsr(), rhs).reshape(x2_nodes, x1_nodes)

max_it = 100
is_cavitated = np.array(pressure.flatten() < 0)
is_cavitated = is_cavitated.astype(float)
test= np.array(is_cavitated)
test = test.astype(float)


for k in range(1,max_it+1):

    M = A @ diags_array(np.ones_like(is_cavitated)-is_cavitated) - B @ diags_array(is_cavitated)
    z = spsolve(M.tocsr(), rhs).reshape(x2_nodes, x1_nodes)
    comparison = np.array(z.flatten() < 0)
    is_cavitated += comparison.astype(float)
    test += comparison.astype(float)
    is_cavitated = np.mod(is_cavitated, 2)
    
print(test)


fig, ax = plt.subplots(figsize=(10, 6))
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}

ax.plot(X1_vector, z[x2_nodes // 2, :], label="Cartesian FVM")


ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()
plt.show()


