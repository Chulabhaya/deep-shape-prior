import torch
from model import DeepShapePrior

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_var = torch.randn(256, 4, 4) # (C, H, W) rather than (H, W, C) for PyTorch
input_var = torch.unsqueeze(input_var, 0) # To add extra batch dimension at beginning
print("Input size: {size}".format(size=input_var.shape))
input_var = input_var.to(device)

model = DeepShapePrior()
model = model.to(device)

output_var = model(input_var)
print("Output size: {size}".format(size=output_var.shape))
