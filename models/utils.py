import torch
import torch.nn as nn

from functools import reduce
from operator import mul

class Resize(nn.Module):
    def __init__(self, shape):
        super(Resize, self).__init__()
        self.shape = shape
        
    def forward(self, x):
        return nn.functional.interpolate(x, size=self.shape, mode='bilinear', align_corners=True)

class LevelAdapter(nn.Module):
    def __init__(self, mapping, shape):
        super(LevelAdapter, self).__init__()
        self.shape = shape
        self.mapping = mapping
        
    def forward(self, x):
        layers = []
        d, w, h = self.shape
        for i in range(d):
            if(i in self.mapping):
                idx = self.mapping.index(i)
                layers.append(x[:, idx:idx+1])
            else:
                zero = torch.zeros(x.size(0), 1, x.size(2), x.size(3)) #requires_grad=True
                layers.append(zero)
        x = torch.cat(layers, dim=1)
        return x