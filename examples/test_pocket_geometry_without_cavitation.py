import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve
from reynolds import film, assembly, boundary


length = 20e-3
width = 500e-3
H_min = 1
H_max = 10
x1_nodes = 64
x2_nodes = 64
a=0.1
b=0.2
h_taper=5e-6
h_min = 20e-6

dX2 = (width / length) / (x2_nodes - 1)
dX1 = 1 / (x1_nodes - 1)

X1_vector = np.linspace(0, 1, x1_nodes)
X2_vector = np.linspace(0, width / length, x2_nodes)

X1, X2 = np.meshgrid(X1_vector, X2_vector)

H = film.cartesian(length, width, h_taper, h_min)
H = film.pocket(H_min, H_max, a, b)(length, width, h_taper, h_min)

H_east, H_west, H_north, H_south = film.faces(H, X1, X2, dX1, dX2)


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
d = B.toarray()
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


def newton_iteration(A_sparse, B_sparse, f, max_iter=300, tol=1e-8):
    n = A_sparse.shape[0]
    z = np.ones(2 * n)  # Initial guess [x; y]
    
    def fischer_burmeister_and_derivative(a, b, eps=1e-12):
        denom = np.sqrt(a**2 + b**2 + eps)
        fb = denom - a - b
        # Derivatives
        da = a / denom - 1.0
        db = b / denom - 1.0
        return fb, da, db

    for iteration in range(max_iter):
        x = z[:n]
        y = z[n:]
        
        # 1. Compute Residual F(z)
        linear_equation = A_sparse @ x - B_sparse @ y - f
        fb, da, db = fischer_burmeister_and_derivative(x, y)
        res = np.concatenate([linear_equation, fb])
        
        # Check convergence
        res_norm = np.max(np.abs(res))
        if res_norm*10 < tol:
            print(f"Converged in {iteration} iterations.")
            break
            
        # Build Exact Sparse Jacobian J(z)
        Da = sp.diags(da)
        Db = sp.diags(db)
        J = sp.bmat([
            [A_sparse, -B_sparse],
            [Da,       Db      ]
        ]).tocsr()
        
        # Solve J * delta_z = -res using a sparse direct solver
        delta_z = spsolve(J, -res)
        
        # Update solution 
        z += delta_z
    else:
        print("Warning: Manual Newton loop reached max iterations.")
        
    x_sol = np.maximum(0, z[:n])
    y_sol = np.maximum(0, z[n:])
    return x_sol, y_sol

pressure, cavitation = newton_iteration(A, B, rhs)



# pressure = spsolve(A.tocsr(), rhs).reshape(x2_nodes, x1_nodes)
# cavitation = spsolve(B.tocsr(), rhs).reshape(x2_nodes, x1_nodes)

# max_it = 100
# is_cavitated = np.array(pressure.flatten() < 0)
# is_cavitated = is_cavitated.astype(float)
# test= np.array(is_cavitated)
# test = test.astype(float)


# for k in range(1,max_it+1):

#     M = A @ sp.diags_array(np.ones_like(is_cavitated)-is_cavitated) - B @ sp.diags_array(is_cavitated)
#     z = spsolve(M.tocsr(), rhs).reshape(x2_nodes, x1_nodes)
#     comparison = np.array(z.flatten() < 0)
#     is_cavitated += comparison.astype(float)
#     test += comparison.astype(float)
#     is_cavitated = np.mod(is_cavitated, 2)
    
# print(test)
pressure = pressure.reshape(x2_nodes, x1_nodes)
cavitation = cavitation.reshape(x2_nodes, x1_nodes)

fig, ax = plt.subplots(figsize=(10, 6))
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}

ax.plot(X1_vector, pressure[x2_nodes // 2, :], label="Cartesian FVM")


ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()
plt.show()


