import torch
import torch.nn.functional as F

# Helper function. Computes the length of the edge connecting nearby points in a 'geometry image'
def compute_edge_length(t):
    _, _, w, h = t.size()
    border = 6

    w0 = torch.Tensor([[-1, 1]]).unsqueeze(0).unsqueeze(1).cuda()
    w1 = torch.Tensor([[-1], [1]]).unsqueeze(0).unsqueeze(1).cuda()

    dx_0 = F.conv2d(t[:, 0, :, :].unsqueeze(1), w0, padding=0)
    dy_0 = F.conv2d(t[:, 1, :, :].unsqueeze(1), w0, padding=0)
    dz_0 = F.conv2d(t[:, 2, :, :].unsqueeze(1), w0, padding=0)

    dx_1 = F.conv2d(t[:, 0, :, :].unsqueeze(1), w1, padding=0)
    dy_1 = F.conv2d(t[:, 1, :, :].unsqueeze(1), w1, padding=0)
    dz_1 = F.conv2d(t[:, 2, :, :].unsqueeze(1), w1, padding=0)

    d0 = dx_0.pow(2) + dy_0.pow(2) + dz_0.pow(2)
    d1 = dx_1.pow(2) + dy_1.pow(2) + dz_1.pow(2)

    d0 = d0.view(d0.size()[0], d0.size()[1], -1)
    d1 = d1.view(d1.size()[0], d1.size()[1], -1)

    ds = torch.cat([d0, d1], dim=2)

    return ds

#This is the regularizer.
#tensors is a list of torch.Tensor objects.
#Each of those tensors is a 'geometry image': an image whose channels are xyz point coordinates.
#We call a helper function compute_edge_length that computes the length of the edges connecting
#nearby points.
def edge_loss(tensors):
    #tensor shape: (batch x (3|6) x img_width x img_height)
    #img_width and img_height are the same
    npatches = len(tensors)
    edge_lengths = []

    edge_lengths.append(compute_edge_length(tensors[:, 0:3, :, :]))

    edge_lengths = torch.cat(edge_lengths, dim=2)

    return edge_lengths.mean() + edge_lengths.std()