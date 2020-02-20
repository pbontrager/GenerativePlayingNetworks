import torch
import torch.nn as nn

import models.utils as utils

import models.deconv_gen as deconv
import models.nearest_gen as nearest
import models.pixelshuffle_gen as pixel

class Generator(nn.Module):
    def __init__(self, latent_shape, env, upsample, dropout, lr):
        super(Generator, self).__init__()

        self.z_size = latent_shape[0]
        if(upsample == 'deconv'):
            self.gen_lvl = deconv.Generator(env.mapping, env.model_shape, latent_shape, dropout)
        elif(upsample == 'pixel'):
            self.gen_lvl = pixel.Generator(env.mapping, env.model_shape, latent_shape, dropout)
        elif(upsample == 'nearest'):
            self.gen_lvl = nearest.Generator(env.mapping, env.model_shape, latent_shape, dropout)
        else:
            raise Exception("Upsample Mode not Implemented")
        self.lvl_to_state = utils.LevelAdapter(env.mapping, env.state_shape)

        self.optimizer = torch.optim.Adam(self.gen_lvl.parameters(), lr = lr)

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
