import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve
from reynolds import film, assembly, boundary


length = 20e-3
width = 2000e-3
H_min = 1
H_max = 10
x1_nodes = 161
x2_nodes = 21
a=0.1
b=0.25
h_taper=5e-6
h_min = 1e-6
cavitation_pressure = -1e5 / 1.2e9 # (p_c - p_in) / (6 mu U l / h0^2)
mu = 0.01
bulk_modulus = 1e8



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

# U=1
# c = bulk_modulus * h_min ** 2 / (6 * mu * length * U) 

# a_P = c * a_P - 0.5 * (H_west - H_east).flatten()
# a_west = a_west + 0.5 * H_west.flatten()
# a_east = a_east - 0.5 * H_east.flatten()


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


b_P = H_east.flatten() * dX2
b_west = H_west.flatten() * dX2
b_east = np.zeros_like(b_west)
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


u_edge = np.exp(-cavitation_pressure/bulk_modulus)-1
rhs = -assembly.identity_rhs(source, fixed)


def newton_iteration(A_sparse, B_sparse, f, max_iter=300, tol=1e-8):
    n = A_sparse.shape[0]
    P0 = spsolve(A_sparse.tocsr(),f)

    z = np.concat([P0, np.zeros( n)])  # Initial guess [x; y]
    
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

pressure, cavitation = newton_iteration(A, B, rhs - A @ (cavitation_pressure*np.ones(x1_nodes*x2_nodes)))


# u = u.reshape(x2_nodes, x1_nodes)
cavitation = cavitation.reshape(x2_nodes, x1_nodes)
pressure = pressure.reshape(x2_nodes, x1_nodes) + cavitation_pressure
# pressure = np.log(u + 1) * bulk_modulus + cavitation_pressure


fig, ax = plt.subplots(figsize=(10, 6))
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}

ax.plot(X1_vector, pressure[x2_nodes // 2, :], label="Cartesian FVM")


ax.set_xlabel("Dimensionless sliding direction")
ax.set_ylabel("Dimensionless pressure")
ax.legend()
plt.show()







# # Analytical comparison
# mu = 0.01      
# U = 1.0       
# p_in = 1e5     # inlet = outlet pressure [Pa]
# p_c = 0.0      # cavitation pressure [Pa]
# beta = 1e12    # bulk modulus [Pa], large beta = incompressible limit 

# h0 = H_min * h_min
# h1 = H_max * h_min
# l = length
# x_a = a * length
# x_b = b * length
# p_scale = 6 * mu * U * length / h0**2

# expected = (p_c - p_in) / p_scale
# if not np.isclose(cavitation_pressure, expected):
#     print(f"Warning: cavitation_pressure should be {expected:.4e} for Table 5.3")


# def analytic_pocket(x, beta):
#     k0 = 6 * mu * U / (beta * h0**2)
#     k1 = 6 * mu * U / (beta * h1**2)
#     # expm1 keeps the small exponents accurate when beta is large
#     C1 = -np.expm1((p_in - p_c) / beta) / np.expm1(-k0 * x_a)       
#     C2 = -C1 * np.exp(-k0 * x_a)                                     
#     C = -C1
#     C3 = h0 / h1 * (1 - C) - 1                                       
#     C5 = -C                                                          
#     C6 = (np.expm1((p_in - p_c) / beta) - C5) / np.exp(k0 * l)       
#     C4 = (-C3 + C5 + C6 * np.exp(k0 * x_b)) / np.exp(k1 * x_b)       
#     x_r = np.log(-C3 / C4) / k1                                      

#     u = np.select(
#         [x <= x_a, x <= x_r, x <= x_b],
#         [C1 + C2 * np.exp(k0 * x), np.zeros_like(x), C3 + C4 * np.exp(k1 * x)],
#         C5 + C6 * np.exp(k0 * x),
#     )
#     return p_c + beta * np.log1p(u), x_r


# x = X1_vector * length
# p_numerical = pressure[x2_nodes // 2, :] * p_scale
# p_analytical, x_r = analytic_pocket(x, beta)

# error = np.sum(np.abs(p_analytical - p_numerical)) / np.sum(p_analytical)
# print(f"reformation at {x_r * 1e3:.3f} mm")
# print(f"p_max analytical {p_analytical.max() / 1e3:.1f} kPa, numerical {p_numerical.max() / 1e3:.1f} kPa")
# print(f"relative error {error:.2e}")

# x_fine = np.linspace(0, length, 2001)
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(x_fine * 1e3, analytic_pocket(x_fine, beta)[0] / 1e6, label=f"Analytical, beta = {beta:.0e}")
# ax.plot(x * 1e3, p_numerical / 1e6, ".", label="Numerical")
# ax.set_xlabel("x [mm]")
# ax.set_ylabel("p [MPa]")
# ax.legend()
# plt.show()

