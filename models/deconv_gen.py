#Simple StyleGAN vs simple DCGAN

import torch
import torch.nn as nn

from functools import reduce
from operator import mul

class Generator(nn.Module):
    def __init__(self, mapping, shapes, z_shape):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        filters = 64

        self.init_shape = (filters, *shapes[0])
        self.preprocess = nn.Sequential(
            nn.Linear(self.z_size, reduce(mul, self.init_shape)),
            nn.ReLU(True))

        self.blocks = nn.ModuleList()
        in_ch = filters
        for s in shapes[1:-1]:
            out_ch = in_ch // 2
            block = nn.Sequential(
                nn.ConvTranspose2d(filters, filters, 3, padding=1),
                nn.Conv2d(filters, filters, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )
            in_ch = out_ch
            self.blocks.append(block)

        out_ch = len(mapping)
        self.output = nn.Sequential(
            utils.Resize(shapes[-1]),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
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
