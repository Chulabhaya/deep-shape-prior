import numpy as np
from pathlib import Path
from sklearn.feature_extraction import image

# Function for adding random noise to point cloud
def add_noise(pc, sigma=0.01, seed=322):
    temp_rand = np.random.RandomState(seed)
    return pc + temp_rand.randn(*pc.shape)*sigma

# Function for saving single PC as .obj file
def save_pc_as_obj(save_path, pc):
    # Bunch of reshaping to match input .obj file format
    pc = pc.cpu().numpy()
    pc = np.transpose(pc)
    pc = np.squeeze(pc)
    pc = np.reshape(pc, (pc.shape[0] * pc.shape[1], pc.shape[2]))

    # Create and open output file
    fo = open(save_path, "a")

    # Save point cloud as .obj file with vertices
    for v in range(pc.shape[0]):
        line = "v {} {} {}\n".format(pc[v, 0], pc[v, 1], pc[v, 2])
        fo.write(line)

def add_faces_to_obj(save_path, grid_h, grid_w, faces_offset=0):
    # Open existing .obj file
    fo = open(save_path, "a")

    # Add faces (as quadrilaterals)
    indices = np.arange(1,grid_h*grid_w + 1,1)
    indices = indices + faces_offset
    indices = np.reshape(indices, (grid_h,grid_w))

    patches = image.extract_patches_2d(indices, (2, 2))
    patches = np.reshape(patches, (patches.shape[0], patches.shape[1] * patches.shape[2]))
    patches = patches[:, [0, 2, 3, 1]]

    for f in range(patches.shape[0]):
        line = "f {} {} {} {}\n".format(patches[f, 0], patches[f, 1], patches[f, 2], patches[f, 3])
        fo.write(line)

    fo.close()