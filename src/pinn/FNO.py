import torch
import torch.nn as nn
import torch.nn.functional as F


# 2D Spectral Convolution (FNO Layer)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to retain in x-direction
        self.modes2 = modes2  # Number of Fourier modes to retain in y-direction

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute 2D Fourier transform (FFT)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space via inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)



# 2D FNO Architecture (The PINO Model)

class FNO(nn.Module):
    def __init__(self, modes=24, width=128):
        super(FNO, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        
        # Input: Channel 1 = Height H(x,y), Channel 2 = Sliding parameter Lambda (broadcasted)
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1) # Output: Pressure P(x,y)

    def forward(self, x, lambd):
        # x shape: [Batch, H, W, 1] (Height grid)
        # lambd shape: [Batch, H, W, 1] (Parameter field)
        grid = torch.cat([x, lambd], dim=-1)
        x = self.fc0(grid)
        x = x.permute(0, 3, 1, 2) # [Batch, Channels, H, W]

        # FNO blocks with GELU activations
        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)

        x1 = self.conv1(x) + self.w1(x)
        x = F.gelu(x1)

        x1 = self.conv2(x) + self.w2(x)
        x = F.gelu(x1)
        
        x1 = self.conv3(x) + self.w3(x)
        x = F.gelu(x1)

        x = x.permute(0, 2, 3, 1) # [Batch, H, W, Channels]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # Output pressure [Batch, H, W, 1]
        return x


# Helper for periodic central differences
def d_dx(f):
    return (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2.0 * dx)

def d_dy(f):
    return (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2.0 * dy)

# Periodic Physics Loss 

def compute_reynolds_residual(H, P, Lambda, dx, dy):

    

    # Compute gradients of Pressure
    dP_dx = d_dx(P)
    dP_dy = d_dy(P)

    # Compute cubic film thickness term: H^3 * grad(P)
    flux_x = H ** 3 * dP_dx
    flux_y = H ** 3 * dP_dy

    # Compute divergence of the flux: div(H^3 * grad(P))
    div_flux = d_dx(flux_x) + d_dy(flux_y)

    # Compute right-hand side source term: Lambda * (dH / dx)
    dH_dx = d_dx(H)
    rhs = Lambda * dH_dx

    # PDE Residual
    residual = div_flux - rhs
    return torch.mean(residual ** 2)



# data driven loss
def data_residual(P_pred,P_num):
    residual = P_pred - P_num
    return torch.mean(residual ** 2)

def derivative_residual(P_pred,P_num,dx,dy):
    # def d_dx(f):
    #     return (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2.0 * dx)
    
    # def d_dy(f):
    #     return (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2.0 * dy)
    
    dx_res = d_dx(P_num) - d_dx(P_pred)
    dy_res = d_dy(P_num) - d_dy(P_pred)
    
    return torch.mean(dx_res ** 2) + torch.mean(dy_res ** 2)
    
    



# Training Step Example
def train_step(model, optimizer, H_batch, Lambda_batch, dx, dy, P_num):
    lambda1 = 1e4
    lambda2 = 1e-2
    lambda3 = 1e4
    
    model.train()
    optimizer.zero_grad()

    # Forward pass: predict pressure field P from height H and lambda
    P_pred = model(H_batch, Lambda_batch)

    # Calculate physics residual loss (unsupervised PINO loss)
    # H_batch shape: [B, H, W, 1], P_pred shape: [B, H, W, 1]
    loss_PDE = compute_reynolds_residual(H_batch.squeeze(-1), P_pred.squeeze(-1), Lambda_batch.squeeze(-1), dx, dy)
    # loss_data = data_residual(P_pred, P_num)
    loss_dxdy = derivative_residual(P_pred, P_num, dx, dy)
    
    loss = lambda2 * loss_PDE + lambda3 * loss_dxdy #+ lambda1 * loss_data 
    loss.backward()
    optimizer.step()
    return loss.item()





if __name__ == "__main__":
    from surfaces import roughness 
    import matplotlib.pyplot as plt
    import numpy as np
    from reynolds import homogenised
    
    import torch

    # Automatically detect ROCm/CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move your model to the GPU
    model = FNO().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

    

    CELL_PROBLEMS = {
        "psi1": (1, (1, 0), -1),
    }
    
    epochs = 15_000
    nodes = 128
    dx = 1 / (nodes - 1)
    dy = 1 / (nodes - 1)
    
    
    
    x = np.linspace(0,1,nodes)
    y = np.linspace(0,1,nodes)
    X,Y = np.meshgrid(x,y)
    
    train_losses = np.zeros(epochs)
    
    batch_size = 8
    Lambda_batch = torch.ones(batch_size,nodes,nodes,1)
    
    for i in range(epochs):
        hurst = np.random.rand(batch_size) * 2
        anisotropy = np.random.rand(batch_size) * 5
        sq = np.random.rand(batch_size) * 0.5
        seed = np.random.rand(batch_size) * 10000
        
        H_batch = torch.ones(batch_size,nodes,nodes,1)
        P_num_batch = torch.ones(batch_size,nodes,nodes,1)
        
        # creating batches
        for j in range(batch_size):
            z = roughness.generate(seed=seed[0], hurst=hurst[j], anisotropy=anisotropy[j], nodes=nodes)
            H_r = roughness.periodic(z, sq=sq[j])
            Hs = H_r(X,Y) + 4
            
            H_batch[j,:,:,0] = torch.from_numpy(Hs).float()
        
    
            solutions, _ = homogenised.solve_cell(roughness=H_r, alpha=4, problems=("psi1",), n=nodes)
            P_num = torch.from_numpy(solutions["psi1"]).float()
            P_num_batch[j,:,:,0] = P_num
            
            # Move to GPU
            H_batch = H_batch.to(device)
            Lambda_batch = Lambda_batch.to(device)
            P_num_batch = P_num_batch.to(device)
            
        train_losses[i] = train_step(model, optimizer, H_batch, Lambda_batch, dx, dy, P_num_batch)
        
        if i % 100 == 0:
            print(train_losses[i])
            
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(epochs), train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
        
    
    
    nodes = 128
    
    
    x_test = np.linspace(0, 1, nodes)
    y_test = np.linspace(0, 1, nodes)
    
    
    X_grid, Y_grid = np.meshgrid(x_test, y_test)
    
    z = roughness.generate(seed=10, hurst=0.5, anisotropy=1, nodes=nodes)
    H_r = roughness.periodic(z, sq=0.15)
    Hs = H_r(X_grid,Y_grid) + 4
    H_batch = torch.from_numpy(Hs).float().unsqueeze(0).unsqueeze(-1).to(device)
    Lambda_batch = torch.ones(nodes, nodes).unsqueeze(0).unsqueeze(-1).to(device)
    
    
        
    with torch.no_grad():
            P_pred = model(H_batch, Lambda_batch)
        
        # Reshape the prediction back into a 2D grid matching X_grid and Y_grid
    P_grid = P_pred.reshape(nodes, nodes).cpu().numpy()
    
    
    # Numerical solution
    
    solutions, _ = homogenised.solve_cell(roughness=H_r, alpha=4, problems=("psi1","psi2"), n=nodes)
    P_num = solutions["psi1"]
    
    
    
    # 3. Plot using matplotlib contourf (filled contour plot)
    plt.axes(projection='3d').plot_surface(X_grid, Y_grid, P_grid-P_grid[0,0], cmap='viridis')
    # plt.figure()#figsize=(8, 6))
    # contour = plt.contourf(X_grid, Y_grid, P_grid-P_grid[0,0], levels=50, cmap='viridis')
    # plt.colorbar(contour, label='Predicted Pressure (P)')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    plt.title('2D FNO Pressure Prediction Field')   
    
    
    plt.axes(projection='3d').plot_surface(X_grid, Y_grid, P_num, cmap='viridis')
    # plt.figure()#figsize=(8, 6))
    # contour = plt.contourf(X_grid, Y_grid, P_num, levels=50, cmap='viridis')
    # plt.colorbar(contour, label='Numerical Pressure (P)')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    plt.title('2D Numerical Pressure Field')
    RMSE = np.mean((P_grid-P_num)**2) ** 0.5
    print("RMS error: ", RMSE)
    
    plt.show()



