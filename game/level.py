import os
import csv
import tempfile
import numpy as np
import timeout_decorator

class Level:
    def __init__(self, generator, ascii_map):
        self.generator = generator
        #self.adapter = adapter
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, 'level.txt')
        
        self.map = ascii_map
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
            l.write(lvl_str[0])
        return state

    @timeout_decorator.timeout(1)
    def test(self, env):
        env.reset()
        env.step(0)
        env.reset()