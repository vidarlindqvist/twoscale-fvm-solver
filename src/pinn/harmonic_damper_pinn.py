import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_width):
        super(MLP, self).__init__()
        
        # Creates the hidden layers. Starts with one input (t) and ends with 
        # one output (y)
        layers = [
            nn.Linear(1, hidden_width),
            nn.Tanh()
            ]
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_width, 1))
        
        

        self.network = nn.Sequential(*layers)
        
    def forward(self, t):
        return self.network(t)
        
        
def physics_res(model, t, params): # t: ncol x 1
    # unpack parameters
    m, mu, k = params
    
    # evaluate NN
    y = model(t)
    
    # compute derivatives
    dydt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    
    d2yd2t = torch.autograd.grad(dydt, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    
    # residual
    return d2yd2t * m + mu * dydt + k * y
        
def boundary(model, tbc):   # size nbc x 1
    
    # evaulate model
    ybc = model(tbc)
    
    # evaluate derivative
    dydt = torch.autograd.grad(ybc, tbc, grad_outputs=torch.ones_like(ybc), create_graph=True)[0]
    
    # return residual 
    return ybc - 1, dydt    # y(0) = 1, y'(0) = 0

def datapoints():
    tbc = torch.zeros(1, 1, requires_grad = True)
    tcolacation = torch.linspace(0, 1, 50, requires_grad = True).reshape(50, 1)
    
    return tbc, tcolacation


def train(tbc, tcolocation, params, model, optimizer):
    
    # set training mode
    model.train()
    
    # zero out gradients
    optimizer.zero_grad()
    
    # compute boundary condition loss
    bc1, bc2 = boundary(model, tbc)
    loss_bc1 = torch.mean(bc1 ** 2)
    loss_bc2 = torch.mean(bc2 ** 2)
    
    # compute physics loss
    r_colocation = physics_res(model, tcolocation, params)
    loss_col = torch.mean(r_colocation ** 2)
    
    # total loss
    lambda1 = 1e-1; lambda2 = 1e-4  # weights
    loss = loss_bc1 + lambda1 * loss_bc2 + lambda2 * loss_col
    
    # backpropogate
    loss.backward()
    
    # update parameters
    optimizer.step()

    # return training loss
    return loss.item()


def exact(t, params):
    m, mu, k = params
    
    delta = mu / (2 * m)
    omega0 = np.sqrt(k / m)
    
    omega = np.sqrt(omega0 ** 2 - delta ** 2)
    phi = np.arctan(-delta/omega)
    A = 1/(2*np.cos(phi))
    y = np.exp(-delta * t) * 2 * A * np.cos(phi + omega * t)
    return y

        
if __name__ == "__main__":
    # set parameters
    m = 1; mu = 4; k = 400
    params = (m, mu, k)
    
    # setup data
    hidden_layers = 4
    hidden_width = 32
    model = MLP(hidden_layers, hidden_width)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    
    # generate data/evaluation points
    tbc, tcolacation = datapoints()
    
    epochs = 4_100
    train_losses = np.zeros(epochs)
    
    for t in range(epochs):
        train_losses[t] = train(tbc, tcolacation, params, model, optimizer)
        
        if t % 100== 0:
            print(train_losses[t])
            
    # import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(epochs), np.log(train_losses))
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    
    t_test = torch.linspace(0, 1, 200).reshape(-1, 1)
    y_hat = model(t_test).detach()
    y_exact = exact(t_test, params)
    
    plt.figure()
    plt.plot(t_test, y_hat, "--")
    plt.plot(t_test, y_exact)
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend(["PINN Prediction", "Exact solution"])
    
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        