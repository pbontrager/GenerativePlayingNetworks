import torch
import torch.nn as nn

from functools import reduce
from operator import mul

#First test on Aliens: ascii -> [".","0","1","2","A"], shape -> (11 x 30): (3, 4) -> (6 x 8) -> (12 x 16) -> (12 x 32)
class Resize(nn.Module):
    def __init__(self, shape):
        super(Resize, self).__init__()
        self.shape = shape
        
    def forward(self, x):
        return nn.functional.interpolate(x, size=self.shape, mode='bilinear', align_corners=True)

class LevelAdapter(nn.Module):
    def __init__(self, mapping, depth):
        super(LevelAdapter, self).__init__()
        self.depth = depth
        self.mapping = mapping
        
    def forward(self, x):
        layers = []
        for i in range(self.depth):
            if(i in self.mapping):
                idx = self.mapping.index(i)
                layers.append(x[:, idx:idx+1])
            else:
                zero = torch.zeros(x.size(0), 1, x.size(2), x.size(3)) #requires_grad=True
                layers.append(zero)
        x = torch.cat(layers, dim=1)
        return x
    
class Generator(nn.Module):
    def __init__(self, mapping, shapes, z_size):
        super(Generator, self).__init__()
        #Simple StyleGAN vs simple DCGAN
        #Output: GVGAI ASCII -> one hot encoding
        self.z_size = z_size
        filters = 512

        self.init_shape = (filters, shapes[0][0], shapes[0][1])
        self.preprocess = nn.Sequential(
            nn.Linear(z_size, reduce(mul, self.init_shape, 1)),
            nn.ReLU(True))

        self.blocks = nn.ModuleList()
        in_ch = filters
        for s in shapes[1:-1]:
            out_ch = in_ch // 2
            block = nn.Sequential(
                Resize(s),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )
            in_ch = out_ch
            self.blocks.append(block)

        out_ch = len(mapping)
        self.output = nn.Sequential(
            Resize(shapes[-1]),
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
                Resize(s),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(True)
            )
            in_ch = out_ch
            out_ch = in_ch * 2
            self.blocks.append(block)
            
        self.postprocess = nn.Sequential(
            Resize(shapes[-1]),
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