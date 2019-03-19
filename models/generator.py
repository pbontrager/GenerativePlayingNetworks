import torch
import torch.nn as nn

import models.utils as utils
import models.bilinear_gen as gen

class Generator(nn.Module):
    def __init__(self, mapping, net_shapes, in_shape, out_shape):
        super(Generator, self).__init__()
        #GAN Network
        self.z_size = in_shape[0]
        self.gen_lvl = gen.Generator(mapping, net_shapes, in_shape)
        self.lvl_to_state = utils.LevelAdapter(mapping, out_shape)

    def forward(self, x):
        return self.gen_lvl(x)

    def new(self, batch):
        with torch.no_grad():
            z = torch.Tensor(batch, self.z_size).uniform_(0, 1)
            lvl = self.forward(z)
            state = self.adapter(lvl)
        return lvl, state

    def adapter(self, x):
        return self.lvl_to_state(x) 