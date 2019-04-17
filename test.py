import torch
from torch.autograd import Variable

n = 200
mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
mu2 = 1 / (n * torch.ones(200))
print('Test')