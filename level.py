import os
import csv
import tempfile
import numpy as np
import timeout_decorator

class Level:
    def __init__(self, generator, lvl_map):
        self.generator = generator
        self.adapter = adapter
        self.path = os.path.join(tempfile.TemporaryDirectory(), 'level.txt')
        
        self.map = lvl_map
        self.map_level = np.vectorize(lambda x: self.map[x])

    def create_levels(self, tensor):
        lvl_array = tensor.argmax(dim=1).cpu().numpy()
        lvls = self.map_level(lvl_array).tolist()
        lvl_strs = ['\n'.join([''.join(row) for row in lvl]) for lvl in lvls]
        return lvl_strs

    def new_level(self):
        lvl_tensor, state = self.generator.new(1)
        lvl_str = self.create_levels(lvl_tensor)
        with open(self.path, 'w') as l:
            l.write(lvl[0])
        return state

    @timeout_decorator.timeout(1)
    def test(self, env):
        self.env.reset()
        self.env.step(0)
        self.env.reset()

# class Generator(nn.Module):
#     def __init__(self):
#         #GAN Network

#     def forward(self, x):
#         #GAN forward

#     def new(self, ammount):
#         with torch.no_grad():
#             z = torch.Tensor(ammount, *size).uniform(0, .1)
#             lvl = self.forward(z)
#             state = self.adapter(lvl)
#         return lvl, state

#     def adapter(self, x):
#         #Adapter code