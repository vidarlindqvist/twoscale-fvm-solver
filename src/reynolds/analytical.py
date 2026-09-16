
# Input data
l = 500e-3
b = 500e-3
delta_h = 5e-6
h0 = 20e-6
delta_H = delta_h / h0

# Real height function
def h(x, y):
    return h0 + delta_h * (l - x) / l

# Dimensionless height function
def H(x, y):
    return h(x * l, y * b) / h0

# 1D Analytical solution
def pressure_1D(Xs):
    return 1 / (delta_H * H(Xs, 0)) - (1 + delta_H) / (delta_H * (2 + delta_H) * H(Xs, 0) ** 2)- 1/(delta_H*(2 + delta_H))