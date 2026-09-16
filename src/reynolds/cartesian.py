import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags

np.set_printoptions(linewidth=200)

# Input data
l = 500e-3
b = 500e-3
delta_h = 5e-6
h0 = 20e-6
delta_H = delta_h / h0

# Grid size
x_nodes = 50
y_nodes = 50

# Grid
X_vector = np.linspace(0, 1, x_nodes)
Y_vector = np.linspace(b/l, 0, y_nodes)
X, Y = np.meshgrid(X_vector, Y_vector)

# Dimensionless increments
delta_X = 1 / (x_nodes - 1)
delta_Y = (b / l) / (y_nodes - 1)

# Real height function
def h(x, y):
    return h0 + delta_h * (l - x) / l

# Dimensionless height function
def H(x, y):
    return h(x * l, y * b) / h0

# Height at intermediate points
H_east = H(X + delta_X / 2, Y)
H_west = H(X - delta_X / 2, Y)
H_north = H(X, Y + delta_Y / 2)
H_south = H(X, Y - delta_Y / 2)

# Coefficients for FVM formulation
# a_P*P_P = a_E+...+a_S+rhs
a_east = 1 / delta_X**2 * H_east.flatten() ** 3
a_west = 1 / delta_X**2 * H_west.flatten() ** 3
a_north = 1 / delta_Y**2 * H_north.flatten() ** 3 
a_south = 1 / delta_Y**2 * H_south.flatten() ** 3 
a_P = a_east + a_west + a_north + a_south
rhs = -((H_east - H_west) / delta_X).flatten()

# Finding the boundaries. 0 if it is on the boundary, 1 if not. 
is_not_east_boundary = np.mod(np.arange(1, x_nodes * y_nodes + 1), x_nodes) > 0
is_not_west_boundary = np.mod(np.arange(x_nodes * y_nodes), x_nodes) > 0
is_not_north_boundary = np.arange(1, x_nodes * y_nodes + 1) <= ((y_nodes - 1) * x_nodes)
is_not_south_boundary = np.arange(1, x_nodes * y_nodes + 1) > x_nodes

is_boundary = np.zeros((y_nodes, x_nodes), dtype = bool)
is_boundary[0,:] = True
is_boundary[:,0] = True
is_boundary[-1,:] = True
is_boundary[:,-1] = True

is_boundary = is_boundary.flatten()

# Building and formatting the diagonals for spdiags
east_diagonal = -np.append([0], (a_east * is_not_east_boundary)[:-1]) * (~is_boundary)
west_diagonal = -np.append((a_west * is_not_west_boundary)[1:], [0]) * (~is_boundary)
north_diagonal = -np.append(np.zeros((x_nodes, )), (a_north * is_not_north_boundary)[:-x_nodes]) * (~is_boundary)
south_diagonal = -np.append((a_south * is_not_south_boundary)[x_nodes:], np.zeros((x_nodes, ))) * (~is_boundary)

# Building the system
diagonals = [a_P, east_diagonal, west_diagonal, north_diagonal, south_diagonal]
A = spdiags(diagonals, [0, 1, -1, x_nodes, -x_nodes], [x_nodes * y_nodes, x_nodes * y_nodes])

# Solving
pressure = np.linalg.solve(A.toarray(), rhs).reshape(y_nodes, x_nodes)

# 1D Analytical solution
def pressure_1D(Xs):
    return 1 / (delta_H * H(Xs, 0)) - (1 + delta_H) / (delta_H * (2 + delta_H) * H(Xs, 0) ** 2)- 1/(delta_H*(2 + delta_H))

Xs = np.linspace(0, 1, 50)
# Plotting

#plt.plot(Xs, pressure_1D(Xs), X_vector, pressure[50,:])
#plt.subplot(projection="3d").plot_surface(*np.indices(pressure.shape), pressure, cmap="viridis")
mesh = plt.pcolormesh(pressure.transpose())
plt.show()






