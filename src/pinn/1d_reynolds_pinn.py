import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_width):
        super(MLP, self).__init__()
        
        # Creates the hidden layers. Start with one input [X, Y, Pbc1, Pbc2]
        layers = [
            nn.Linear(2, hidden_width),
            nn.Tanh()
            ]
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_width, 1))


        self.network = nn.Sequential(*layers)
        
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        return self.network(inputs)



def known_H(X, Y):
    # return 1.1 + torch.sin(2*np.pi*X ) 
    return 0.3 * (2 + torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y))


        
def physics_res(model, x, y): 
    # evaluate NN
    P = model(x, y)
    H = known_H(x, y)
    
    # compute derivatives
    dP_dx = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    dP_dy = torch.autograd.grad(P, y, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    
    dH_dx = torch.autograd.grad(H, x, grad_outputs=torch.ones_like(H), create_graph=True)[0]
    
    flux_x = (H ** 3) * dP_dx
    flux_y = (H ** 3) * dP_dy
    
    dflux_dx = torch.autograd.grad(flux_x, x, grad_outputs=torch.ones_like(flux_x), create_graph=True)[0]
    dflux_dy = torch.autograd.grad(flux_y, y, grad_outputs=torch.ones_like(flux_y), create_graph=True)[0]
    # residual
    return dflux_dx + dflux_dy - dH_dx
        
def boundary_loss(model, grid_size):  
    
    # Grid points
    zeroes = torch.zeros(grid_size, 1, requires_grad=True)
    ones = torch.ones(grid_size, 1, requires_grad=True)
    grid = torch.linspace(0, 1, grid_size).view(-1, 1).requires_grad_(True)
    
    # X-Direction Boundaries (x = 0 and x = 1, y varies along grid)
    Px0 = model(zeroes, grid)
    Px1 = model(ones, grid)
    
    # Derivatives w.r.t x
    dPx0_dx = torch.autograd.grad(Px0, zeroes, grad_outputs=torch.ones_like(Px0), create_graph=True)[0]
    dPx1_dx = torch.autograd.grad(Px1, ones, grad_outputs=torch.ones_like(Px1), create_graph=True)[0]

    # Y-Direction Boundaries (x varies along grid, y = 0 and y = 1)
    Py0 = model(grid, zeroes)
    Py1 = model(grid, ones)
    
    # Derivatives w.r.t y
    dPy0_dy = torch.autograd.grad(Py0, zeroes, grad_outputs=torch.ones_like(Py0), create_graph=True)[0]
    dPy1_dy = torch.autograd.grad(Py1, ones, grad_outputs=torch.ones_like(Py1), create_graph=True)[0]
    
    
    # Losses
    loss_bc_x = torch.mean((Px0 - Px1) ** 2)
    loss_bc_dx = torch.mean((dPx0_dx - dPx1_dx) ** 2)
    
    loss_bc_y = torch.mean((Py0 - Py1) ** 2)
    loss_bc_dy = torch.mean((dPy0_dy - dPy1_dy) ** 2)
    
    # return residual 
    return loss_bc_x + loss_bc_dx + loss_bc_y + loss_bc_dy


def train(model, optimizer, grid_size=64):
    
    # set training mode
    model.train()
    
    # zero out gradients
    optimizer.zero_grad()
    
    # compute boundary condition loss

    loss_bc = boundary_loss(model, grid_size)
    
    x_collocation = torch.rand(grid_size, 1, requires_grad=True)
    y_collocation = torch.rand(grid_size, 1, requires_grad=True)
    
    # x_lin = torch.linspace(0, 1, 8)
    # y_lin = torch.linspace(0, 1, 8)
    # X_grid, Y_grid = torch.meshgrid(x_lin, y_lin, indexing='ij')
    
    # # Flatten them into column vectors of shape (grid_size * grid_size, 1) and enable gradients
    # x_collocation = X_grid.reshape(-1, 1).requires_grad_(True)
    # y_collocation = Y_grid.reshape(-1, 1).requires_grad_(True)
    
    
    # compute physics loss

    r_colocation = physics_res(model, x_collocation, y_collocation)
    loss_col = torch.mean(r_colocation ** 2)
    
    # Locks z-axis
    origin = torch.tensor([[0]], dtype=torch.float32)
    loss_scale = model(origin,origin) ** 2
    
    # Total loss combining physics and soft boundary constraints
    lambda_bc = 1e-2
    lambda_col = 1e-2
    lambda_scale = 1e-2
    
    loss = lambda_bc * loss_bc + lambda_col * loss_col + lambda_scale * loss_scale
    
    # backpropogate
    loss.backward()
    
    # update parameters
    optimizer.step()

    # return training loss
    return loss.item()


def exact(x, y, H_taper):
    H = known_H(x, y, H_taper)
    return (
        1 / (H_taper * H)
        - (1 + H_taper) / (H_taper * (2 + H_taper) * H**2)
        - 1 / (H_taper * (2 + H_taper))
    )

        
if __name__ == "__main__":
    # set parameters
    
    # setup data
    hidden_layers = 4
    hidden_width = 32
    model = MLP(hidden_layers, hidden_width)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    
    
    epochs = 4_000
    train_losses = np.zeros(epochs)
    
    for x in range(epochs):
        train_losses[x] = train(model, optimizer, grid_size=64)
        
        if x % 100== 0:
            print(train_losses[x])
            
    # import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(range(epochs), train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    
    
    x_test = torch.linspace(0, 1, 200)
    y_test = torch.linspace(0, 1, 200)
    
    X_grid, Y_grid = torch.meshgrid(x_test, y_test, indexing='ij')
    x_flat = X_grid.reshape(-1, 1)
    y_flat = Y_grid.reshape(-1, 1)
    
with torch.no_grad():
        P_pred = model(x_flat, y_flat)
    
    # Reshape the prediction back into a 2D grid matching X_grid and Y_grid
P_grid = P_pred.reshape(200, 200).cpu().numpy()

# Convert PyTorch meshgrids to NumPy for plotting
X_np = X_grid.cpu().numpy()
Y_np = Y_grid.cpu().numpy()

# 3. Plot using matplotlib contourf (filled contour plot)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X_np, Y_np, P_grid, levels=50, cmap='viridis')
plt.colorbar(contour, label='Predicted Pressure (P)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D PINN Pressure Prediction Field')
# plt.axes(projection='3d').plot_surface(X_np, Y_np, P_grid, cmap='viridis')
plt.show()
    
