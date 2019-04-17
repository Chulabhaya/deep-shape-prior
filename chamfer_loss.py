import numpy as np
import torch

def chamfer_loss(x, y, subsampling_rate=1.0):
    # Reshape point clouds for later calculations
    x = torch.transpose(x, 1, 3).contiguous()
    y = torch.transpose(y, 1, 3).contiguous()
    x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
    y = y.view(y.size(0), y.size(1) * y.size(2), y.size(3))

    # If sampling rate is 1.0, that's equivalent to doing no subsampling
    # Randomly subsample first point cloud
    x_num_points = x.shape[1]
    num_samples = int(round(1 * subsampling_rate * x_num_points))
    indices = np.arange(x_num_points).astype(int)
    np.random.shuffle(indices)
    indices = torch.from_numpy(indices[:num_samples])
    x = x[:, indices, :]

    # Randomly subsample second point cloud
    y_num_points = y.shape[1]
    num_samples = int(round(1 * subsampling_rate * y_num_points))
    indices = np.arange(y_num_points).astype(int)
    np.random.shuffle(indices)
    indices = torch.from_numpy(indices[:num_samples])
    y = y[:, indices, :]

    # Calculate individual Chamfer distances
    dist1 = nn_dist_l2(x, y)
    dist2 = nn_dist_l2(y, x)
    #dist1 = nn_dist_lp(x, y, 1)
    #dist2 = nn_dist_lp(y, x, 1)

    dist_sum = dist1 + dist2

    return dist_sum

# Very fast, very vectorized approach for calculating L2/Euclidean distance (when x and y are 
# the same sizes)
# Thanks to: https://discuss.pytorch.org/t/fastest-way-to-find-nearest-neighbor-for-a-set-of-points/5938/12
def nn_dist_l2(x, y):
    # Calculates distance as (x - y)^2
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    xy = torch.bmm(x, y.transpose(2,1))

    num_points = xx.shape[1]

    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    distances = rx.transpose(2,1) + ry - 2*xy

    minimums, _ = torch.min(distances, 1)
    chamfer_dist = torch.mean(minimums, 1) 

    return chamfer_dist

# Generalized function for calculating distance between points as
# a form of Minkowski distance
def nn_dist_lp(x, y, p):
    # Calculates distance as (|x - y|)^(p)
    differences = torch.abs(x.unsqueeze(2) - y.unsqueeze(1))
    dif_exp = torch.pow(differences, p)
    distances = torch.sum(dif_exp, 3)

    minimums, _ = torch.min(distances, 1)
    chamfer_dist = torch.mean(minimums, 1)

    return chamfer_dist

