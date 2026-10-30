import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_width):
        super(MLP, self).__init__()
        
        # Creates the hidden layers. Start with one input [X, Pbc1, Pbc2]
        layers = [
            nn.Linear(3, hidden_width),
            nn.Tanh()
            ]
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_width, 1))
        
        

        self.network = nn.Sequential(*layers)
        
    def forward(self, x, Pbc1, Pbc2):
        inputs = torch.cat([x, Pbc1, Pbc2], dim=1)
        return self.network(inputs)

def known_H(X, H_taper):
    return 1 + H_taper * (1 - X)

        
def physics_res(model, x, data): # t: ncol x 1
    Pbc1, Pbc2, H_taper = data
    # evaluate NN
    P = model(x, Pbc1, Pbc2)
    H = known_H(x, H_taper)
    
    # compute derivatives
    dP_dx = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    
    dH_dx = torch.autograd.grad(H, x, grad_outputs=torch.ones_like(H), create_graph=True)[0]
    
    flux = (H ** 3) * dP_dx
    
    dflux_dx = torch.autograd.grad(flux, x, grad_outputs=torch.ones_like(flux), create_graph=True)[0]
    # residual
    return dflux_dx - dH_dx
        
def boundary_loss(model, batch_size):   # size nbc x 1
    
    # Pb = calculated values at boundaries
    # Pbc = boundary conditions
    Pbc1 = torch.rand(batch_size, 1) * 4.0 - 2.0
    Pbc2 = torch.rand(batch_size, 1) * 4.0 - 2.0
    
    # Boundary 1: X = 0 -> Target is Pbc1
    x0 = torch.zeros(batch_size, 1)
    pred_0 = model(x0, Pbc1, Pbc2)
    loss_bc0 = torch.mean((pred_0 - Pbc1) ** 2)
    
    # Boundary 2: X = 1 -> Target is Pbc2
    x1 = torch.ones(batch_size, 1)
    pred_1 = model(x1, Pbc1, Pbc2)
    loss_bc1 = torch.mean((pred_1 - Pbc2) ** 2)
    
    # return residual 
    return loss_bc0 + loss_bc1

def datapoints():
    xbc = torch.tensor([[0.0], [1.0]])
    xcolacation = torch.linspace(0, 1, 50, requires_grad = True).reshape(50, 1)
    
    return xbc, xcolacation


def train(model, optimizer, H_taper,batch_size=64):
    
    # set training mode
    model.train()
    
    # zero out gradients
    optimizer.zero_grad()
    
    # compute boundary condition loss

    loss_bc = boundary_loss(model, batch_size)
    
    x_collocation = torch.rand(batch_size, 1, requires_grad=True)
    Pbc1_col = torch.rand(batch_size, 1) * 4.0 - 2.0
    Pbc2_col = torch.rand(batch_size, 1) * 4.0 - 2.0
    
    # compute physics loss
    
    data = (Pbc1_col, Pbc2_col, H_taper)
    
    
    r_colocation = physics_res(model, x_collocation, data)
    loss_col = torch.mean(r_colocation ** 2)
    
    # Total loss combining physics and soft boundary constraints
    lambda_bc = 1e-1
    lambda_col = 1e-2
    loss = lambda_bc * loss_bc + lambda_col * loss_col
    
    # backpropogate
    loss.backward()
    
    # update parameters
    optimizer.step()

    # return training loss
    return loss.item()


def exact(x, H_taper):
    H = known_H(x, H_taper)
    return (
        1 / (H_taper * H)
        - (1 + H_taper) / (H_taper * (2 + H_taper) * H**2)
        - 1 / (H_taper * (2 + H_taper))
    )

        
if __name__ == "__main__":
    # set parameters
    H_taper = 0.25
    
    # setup data
    hidden_layers = 4
    hidden_width = 32
    model = MLP(hidden_layers, hidden_width)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    
    
    epochs = 4_000
    train_losses = np.zeros(epochs)
    
    for x in range(epochs):
        train_losses[x] = train(model, optimizer, H_taper, batch_size=64)
        
        if x % 100== 0:
            print(train_losses[x])
            
    # import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(range(epochs), train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    
    
    
    
    x_test = torch.linspace(0, 1, 200).reshape(-1, 1)
    Pbc1_test = torch.zeros_like(x_test)
    Pbc2_test = torch.zeros_like(x_test)

    P_hat = model(x_test, Pbc1_test, Pbc2_test).detach()
    P_exact = exact(x_test, H_taper)
    
    plt.figure()
    plt.plot(x_test, P_hat, "--")
    plt.plot(x_test, P_exact)
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend(["PINN Prediction", "Exact solution"])
    
    plt.show()
    
