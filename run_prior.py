import numpy as np
import torch
import torch.optim as optim

from model import DeepShapePrior
from custom_loss_functions import chamfer_loss
from utility import add_noise, save_pc_as_obj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up initial point cloud data
pc_name = 'bunny_0'
data_path = 'data/' + pc_name + '.npy'

clean_pc = np.load(data_path)
clean_pc = np.reshape(clean_pc, (128, 128, 3))
noisy_pc = add_noise(clean_pc, seed=322) # Only local RNG seed set, not global
clean_pc = torch.FloatTensor(clean_pc.transpose()).unsqueeze(0)
noisy_pc = torch.FloatTensor(noisy_pc.transpose()).unsqueeze(0)

# Save clean and noisy point clouds
output_path = 'output/' + pc_name + '_clean' + '.obj'
save_pc_as_obj(output_path, clean_pc)
output_path = 'output/' + pc_name + '_noisy' + '.obj'
save_pc_as_obj(output_path, noisy_pc)

# Initialize noisy input
torch.manual_seed(322)
noisy_input = torch.randn(1, 256, 4, 4)
noisy_input.detach()

# Create model
model = DeepShapePrior()
model = model.to(device)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_iter = 5000
for i in range(num_iter):
    noisy_input = noisy_input.to(device)
    noisy_pc = noisy_pc.to(device) 

    out = model(noisy_input)

    optimizer.zero_grad()
    loss = chamfer_loss(out, noisy_pc, subsampling_rate=0.25)
    loss.backward()
    optimizer.step()

    print("Iteration: {}, Training Loss: {}".format(i, loss.float()))

# Generate and save output from trained model (hopefully de-noised)
denoised_pc = model(noisy_input)
output_path = 'output/' + pc_name + '_denoised' + '.obj'
save_pc_as_obj(output_path, denoised_pc.detach())

print('Finished!')