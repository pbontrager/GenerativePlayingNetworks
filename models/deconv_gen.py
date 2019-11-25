#Simple StyleGAN vs simple DCGAN

import torch
import torch.nn as nn

from functools import reduce
from operator import mul

import models.utils as utils

import pdb

class Generator(nn.Module):
    def __init__(self, mapping, shapes, z_shape):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        layers = len(mapping)

        self.active = nn.ReLU(True)

        #add conv layers between
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.z_size, 256, (3, 4), 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*8) x 3 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf*4) x 6 x 8
            nn.ConvTranspose2d(128, layers, 4, 2, 1, bias=False),
            # state size. (ngf*2) x 12 x 16
        )

        self.output = nn.Softmax2d()

    def forward(self, z):
        x = z.reshape(-1, self.z_size, 1, 1)
        x = self.main(x)
        x = self.output(x)
        return x
