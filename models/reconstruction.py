import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import mul
import math

import models.utils as utils

import pdb

#class BaseCNN(nn.Module)
#CNN -> 3 layer CNN, linear transform to hidden, step through gru for new hidden
#return -> linear pass through critic, action_features, gru_hidden


#class Decoder(nn.Module)
#Encode state to hidden value (same as current CNN)
#Reconstruct from x, x is stil used for linear_critic, and action_features

class Decoder(nn.Module):
    def __init__(self, num_outputs, hidden_size=512, shapes=[]):
        super(Decoder, self).__init__()

        in_ch = 32
        self.initial_shape = (in_ch, *shapes[0])

        self.features = nn.Linear(hidden_size, in_ch * reduce(mul, shapes[0]))

        #[(3, 4),(6, 8),(12, 16)]
        self.blocks = nn.ModuleList()
        halfway = math.ceil(len(shapes)/2)
        for i, s in enumerate(shapes):
            if(i == len(shapes) - 1):
                out_ch = num_outputs
            elif(i+1 == halfway and len(shapes)%2==0):
                out_ch = in_ch
            elif(i+1 < halfway):
                out_ch = in_ch*2
            else:
                out_ch = in_ch//2
            block = nn.Sequential(
                utils.Resize(s),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU())
            in_ch = out_ch
            self.blocks.append(block)

    def forward(self, x):
        #pdb.set_trace()
        print("PDB check the output of this network")
        x = self.features(x)
        x = x.view(-1, *self.initial_shape)
        for b in self.blocks:
            x = b(x)
        return x


#class CapsuleNet(nn.Module)
#Use magnitude to get probabilites, not the "Categorical function
#	Use a custom self.dist -> calculate relative probabilities from magnitudes and pass to pytorch dist
# Calculate the value from linear_critic of output vector (also use for reconstruction)
