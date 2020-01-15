#Simple StyleGAN vs simple DCGAN

import torch
import torch.nn as nn

from functools import reduce
from operator import mul

import models.utils as utils

class Generator(nn.Module):
    def __init__(self, mapping, shapes, z_shape):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        filters = 512

        self.init_shape = (filters, *shapes[0])
        self.preprocess = nn.Sequential(
            nn.Linear(self.z_size, reduce(mul, self.init_shape), bias=False),
            nn.LeakyReLU(True))

        self.blocks = nn.ModuleList()
        in_ch = filters
        for s in shapes[1:-1]:
            out_ch = in_ch // 2
            block = nn.Sequential(
                #utils.Resize(s), #pixel shuffle
                nn.Conv2d(in_ch, 2*out_ch, 3, padding=1, bias=False), #pixel shuffle *2
                nn.LeakyReLU(True),
                nn.Conv2d(2*out_ch, out_ch*4, 3, padding=1, bias=False), #pixel shuffle *4
                nn.BatchNorm2d(out_ch*4), #pixel shuffle
                nn.LeakyReLU(True),
                nn.PixelShuffle(2) #pixel shuffle
            )
            in_ch = out_ch
            self.blocks.append(block)

        out_ch = len(mapping)
        self.output = nn.Sequential(
            #utils.Resize(shapes[-1]), #pixel shuffle
            nn.Conv2d(in_ch, out_ch*4, 3, padding=1, bias=True), #Pixel Shuffle
            nn.PixelShuffle(2) #Pixel Shuffle
            nn.Softmax2d()
        )

    def forward(self, z):
        x = self.preprocess(z)
        d, h, w = self.init_shape
        x = x.view(-1, d, h, w)
        for b in self.blocks:
            x = b(x)
        x = self.output(x)
        return x
