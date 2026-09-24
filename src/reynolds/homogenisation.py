import numpy as np
from scipy.sparse.linalg import spsolve 
from . import assembly, cell, film, boundary

def solve(length=1e-3,
    width=1e-3,
    h_taper=5e-6,
    h_min=20e-6,
    x1_nodes=50,
    x2_nodes=50,
    y1_nodes=50,
    y2_nodes=50,
    roughness_amplitude=1,
    film_thickness=film.cartesian,
    roughness=film.bisinusoidal
            ):

    H_taper = h_taper / h_min
    X1_vector = np.linspace(0, 1, x1_nodes)
    X2_vector = np.linspace(0, width / length, x2_nodes)
    Y1_vector = np.linspace(0, 1, y1_nodes)
    Y2_vector = np.linspace(0, 1, y2_nodes)
   
    X1, X2 = np.meshgrid(X1_vector, X2_vector)
    Y1, Y2 = np.meshgrid(Y1_vector, Y2_vector)
    
    dX1 = 1 / (x1_nodes - 1)
    dX2 = (width / length) / (x1_nodes - 1)
    dY1 = 1 / (y1_nodes - 1)
    dY2 = 1 / (y2_nodes - 1)
   
    H0 = film_thickness(length, width, h_taper, h_min)
    Hr = roughness(roughness_amplitude)
   
    H0_east, H0_west, H0_north, H0_south = film.faces(H0, X1, X2, dX1, dX2)
    Hr_east, Hr_west, Hr_north, Hr_south = film.faces(Hr, Y1, Y2, dY1, dY2)
    a_east, a_west, a_north, a_south, a_P = assembly.cartesian_coefficients(
        H0_east**3, H0_west**3, H0_north**3, H0_south**3, dX1, dX2
    )
    alpha = np.array(film.faces(H0, X1, X2, dX1, dX2))
    phi_x, phi_y, phi_s, phi_xy, phi_yx, phi_sy = cell.flow_factors(roughness, alpha, n=y1_nodes)
   
    source = -assembly.build_source(H0_east, H0_west, H0_north, H0_south, dX1, dX2, e=(1, 0))
   
    fixed = boundary.all_edges(x1_nodes, x2_nodes)
    rhs = assembly.identity_rhs(source, fixed)
   
    A = assembly.five_point(
        a_east, a_west, a_north, a_south, a_P, x1_nodes, x2_nodes, fixed
    )
   
    pressure = spsolve(A.tocsr(), rhs).reshape(x2_nodes, x1_nodes)
   
    return pressure, X1_vector, X2_vector, H_taper
