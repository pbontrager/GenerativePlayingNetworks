import torch
import torch.nn as nn

import models.utils as utils
import models.deconv_gen as gen
#import models.bilinear_gen as gen
#import models.pixelshuffle_gen as gen

class Generator(nn.Module):
    def __init__(self, latent_shape, env):
        super(Generator, self).__init__()
        #GAN Network
        self.z_size = latent_shape[0]
        self.gen_lvl = gen.Generator(env.mapping, env.model_shape, latent_shape)
        self.lvl_to_state = utils.LevelAdapter(env.mapping, env.state_shape)

        self.optimizer = torch.optim.Adam(self.gen_lvl.parameters(), lr = 0.0001)

    def forward(self, x):
        return self.gen_lvl(x)

    def new(self, z):
        self.gen_lvl.eval()
        with torch.no_grad():
            lvl = self.forward(z)
            state = self.adapter(lvl)
        self.gen_lvl.train()
        return lvl, state

    def adapter(self, x):
        return self.lvl_to_state(x)
