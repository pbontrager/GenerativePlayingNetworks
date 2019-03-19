import torch
import torch.nn as nn

from functools import reduce
from operator import mul

import models.utils as utils

class Critic(nn.Module):
    def __init__(self, ascii, shapes):
        super(Critic, self).__init__()
        filters = 64

        #self.init_shape = (filters, shapes[0][0], shapes[0][1])
        #self.preprocess = nn.Sequential(
        #    nn.Linear(z_size, reduce(mul, self.init_shape, 1)),
        #    nn.ReLU(True))

        self.blocks = nn.ModuleList()
        in_ch = len(ascii)
        out_ch = filters
        for s in shapes[1:-1]:
            block = nn.Sequential(
                utils.Resize(s),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(True)
            )
            in_ch = out_ch
            out_ch = in_ch * 2
            self.blocks.append(block)
            
        self.postprocess = nn.Sequential(
            utils.Resize(shapes[-1]),
            nn.AdaptiveAvgPool2d(1)
        )
        self.winner = nn.Sequential(
            nn.Linear(in_ch, 1),
            nn.Sigmoid()
        )
        self.steps = nn.Sequential(
            nn.Linear(in_ch, 1),
            nn.Sigmoid()
        )
        self.compiles = nn.Sequential(
            nn.Linear(in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.postprocess(x)
        x = x.view(x.size(0),-1)
        winner = self.winner(x)
        steps = self.steps(x)
        compiles = self.compiles(x)
        return compiles, winner, steps