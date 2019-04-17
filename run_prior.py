import numpy as np
import torch
import torch.optim as optim

# from full_decoder_model import FullDecoder
from small_decoder_model import SmallDecoder
from chamfer_loss import chamfer_loss
from edge_loss import edge_loss
from utility import add_noise, save_pc_as_obj, add_faces_to_obj

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
# Full decoder code
# model = FullDecoder()
# model = model.to(device)

# Multiple decoder code
num_small_decoders = 4
small_decoders = {}
model_names = []
for d in range(num_small_decoders):
    model_name = "model" + str(d)
    model_names.append(model_name)
    model = SmallDecoder()
    model = model.to(device)
    small_decoders[model_name] = model

# Create optimizer
# Full decoder code
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# Multiple decoder code
model_parameters = []
for d in range(num_small_decoders):
    model_parameters += list(small_decoders[model_names[d]].parameters())
optimizer = optim.Adam(model_parameters, lr=0.01)

# Training loop
num_iter = 5000
for i in range(num_iter):
    noisy_input = noisy_input.to(device)
    noisy_pc = noisy_pc.to(device) 

    # Full decoder code
    # out = model(noisy_input)

    # Multiple decoder code
    all_out = []
    for d in range(num_small_decoders):
        model = small_decoders[model_names[d]]
        out = model(noisy_input)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(out.size(0), out.size(1) * out.size(2), out.size(3))
        all_out.append(out)
    out = torch.cat(all_out, dim=1)
    out = torch.transpose(out, 1, 2).contiguous()
    out = out.view(out.size(0), out.size(1), 128, 128)
    
    optimizer.zero_grad()
    loss = chamfer_loss(out, noisy_pc, subsampling_rate=0.25) + edge_loss(out)
    loss.backward()
    optimizer.step()

    print("Iteration: {}, Training Loss: {}".format(i, loss.float()))

# Generate and save output from trained model (hopefully de-noised)
# Full decoder code
# denoised_pc = model(noisy_input)
# output_path = 'output/' + pc_name + '_denoised' + '.obj'
# save_pc_as_obj(output_path, denoised_pc.detach(), add_faces=True)

# Multiple decoder code
all_out = []
output_path_final = 'output/' + pc_name + '_denoised_final' + '.obj'
output_path_final_points = 'output/' + pc_name + '_denoised_final_points' + '.obj'
for d in range(num_small_decoders):
    model = small_decoders[model_names[d]]
    out = model(noisy_input)

    # Save individual decoder pieces
    output_path = 'output/' + pc_name + '_denoised_' + str(d) + '.obj'
    save_pc_as_obj(output_path, out.detach())
    add_faces_to_obj(output_path, out.shape[2], out.shape[3])

    # Save overall combined point cloud output
    save_pc_as_obj(output_path_final, out.detach())
    save_pc_as_obj(output_path_final_points, out.detach())

    out = torch.transpose(out, 1, 3).contiguous()
    out = out.view(out.size(0), out.size(1) * out.size(2), out.size(3))
    all_out.append(out)

# Add faces to a combined output
for d in range(num_small_decoders):
    add_faces_to_obj(output_path_final, int(np.sqrt(all_out[d].shape[1])), int(np.sqrt(all_out[d].shape[1])), faces_offset=d*all_out[d].shape[1])

print('Finished!')