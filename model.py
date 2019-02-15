import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_function='leaky_relu'):
        super(ConvBlock, self).__init__()
        
        # Each ConvBlock doubles the spatial dimensions of the input, and reduces the 
        # number of channels (generally we halve the number of channels, but it can vary)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if activation_function == 'leaky_relu':
            self.activation_function = nn.LeakyReLU(negative_slope=0.02)
        elif activation_function == 'tanh':
            self.activation_function = nn.Tanh()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.bn(self.conv(x))
        x = self.activation_function(x)
        
        return x

class DeepShapePrior(nn.Module):
    def __init__(self):
        super(DeepShapePrior, self).__init__()

        # This model combines several ConvBlocks such that an input
        # of dimension 256x4x4 (channels = 256, h = 4, w = 4)
        # ends up as 3x128x128 (channels = 3, h = 128, w = 128)
        self.conv_block1 = ConvBlock(256, 128)
        self.conv_block2 = ConvBlock(128, 64)
        self.conv_block3 = ConvBlock(64, 32)
        self.conv_block4 = ConvBlock(32, 16)
        self.conv_block5 = ConvBlock(16, 3, 'tanh')

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        return x