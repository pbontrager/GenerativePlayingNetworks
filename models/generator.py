import torch
import torch.nn as nn

import models.utils as utils
import models.bilinear_gen as gen

class Generator(nn.Module):
    def __init__(self, latent_shape, env):
        super(Generator, self).__init__()
        #GAN Network
        self.z_size = latent_shape[0]
        self.gen_lvl = gen.Generator(env.mapping, env.model_shape, latent_shape)
        self.lvl_to_state = utils.LevelAdapter(env.mapping, env.state_shape)

    def forward(self, x):
        return self.gen_lvl(x)

    def new(self, z):
        with torch.no_grad():
            lvl = self.forward(z)
            state = self.adapter(lvl)
        return lvl, state

    def adapter(self, x):
        return self.lvl_to_state(x)
