import numpy as np
from pathlib import Path

# Function for adding random noise to point cloud
def add_noise(pc, sigma=0.01, seed=322):
    temp_rand = np.random.RandomState(seed)
    return pc + temp_rand.randn(*pc.shape)*sigma

# Function for saving single PC as .obj file
def save_pc_as_obj(save_path, pc):
    pc = pc.cpu().numpy()
    pc = np.transpose(pc)
    pc = np.squeeze(pc)
    pc = np.reshape(pc, (pc.shape[0] * pc.shape[1], pc.shape[2]))

    # Create and open output file
    fo = open(save_path, "w")

    # Save point cloud as .obj file
    for v in range(pc.shape[0]):
        line = "v {} {} {}\n".format(pc[v, 0], pc[v, 1], pc[v, 2])
        fo.write(line)
    
    fo.close()