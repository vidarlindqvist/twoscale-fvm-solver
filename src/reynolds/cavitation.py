"""
This script solves for cavitation in the cartesian case
"""

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve
from reynolds import film, assembly, boundary




def solve(
        length = 20e-3,
        width = 2000e-3,
        x1_nodes = 161,
        x2_nodes = 21,
        cavitation_pressure = -1e-4, # (p_c - p_in) / (6 mu U l / h0^2)
        film_thickness=film.pocket(10, 1, 0.1, 0.25),
        # Flow factors
        phi_x=lambda H: H**3,
        phi_y=lambda H: H**3,
        phi_s=lambda H: H,
        ):
    

    dX2 = (width / length) / (x2_nodes - 1)
    dX1 = 1 / (x1_nodes - 1)
    
    X1_vector = np.linspace(0, 1, x1_nodes)
    X2_vector = np.linspace(0, width / length, x2_nodes)
    X1, X2 = np.meshgrid(X1_vector, X2_vector)
    
    H = film_thickness
    
    
    
    # Building the system
    H_east, H_west, H_north, H_south = film.faces(H, X1, X2, dX1, dX2)
    fixed = boundary.all_edges(x1_nodes, x2_nodes)
    
    a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
        phi_x(H_east),
        phi_x(H_west),
        phi_y(H_north),
        phi_y(H_south),
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
        phi_s(H_east),
        phi_s(H_west),
        phi_s(H_north),
        phi_s(H_south),
        dX1,
        dX2,
        (1,0)
        )
    
    
    rhs = -assembly.identity_rhs(source, fixed)
    
    

    # Solves Ax-By - rhs = 0 using newtons method
    def newton_iteration(A_sparse, B_sparse, f, max_iter=300, tol=1e-8):
        n = A_sparse.shape[0]
        P0 = spsolve(A_sparse.tocsr(),f)
    
        z = np.concat([P0, np.zeros(n)])  # Initial guess [x; y]

        
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
            if res_norm < tol:
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

    cavitation = cavitation.reshape(x2_nodes, x1_nodes)
    pressure = pressure.reshape(x2_nodes, x1_nodes) + cavitation_pressure
    return pressure, cavitation



# fig, ax = plt.subplots(figsize=(10, 6))
# bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
# ax.plot(X1_vector, pressure[x2_nodes // 2, :], label="Cartesian FVM")
# ax.set_xlabel("Dimensionless sliding direction")
# ax.set_ylabel("Dimensionless pressure")
# ax.legend()
# plt.show()









