import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

device = torch.device("cuda")

x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

z = x @ y

print("GPU:", torch.cuda.get_device_name(0))
print("Result:", z.mean().item())
print("GPU computation successful!")