import os
import csv
import numpy as np
import timeout_decorator

import gym
import gym_gvgai

class Game:
    def __init__(self, game, lvl_map):
        self.name = game
        self.map = lvl_map
        self.env = gym_gvgai.make('gvgai-{}-lvl0-v0'.format(game)).unwrapped
        self.map_level = np.vectorize(lambda x: self.map[x])

    def create_levels(self, tensor):
        lvl_array = tensor.argmax(dim=1).cpu().numpy()
        lvls = self.map_level(lvl_array).tolist()
        lvl_strs = ['\n'.join([''.join(row) for row in lvl]) for lvl in lvls]
        return lvl_strs
        
    def load_level(self, lvl, name):
        try:
            with open(name, 'w') as l:
                l.write(lvl)
            self.set_level(name)
            
            #Manual regularization to encourage 1 agent
            #compile_score = min(1 - (lvl.count('A') - 1)/len(lvl), 1)
            compile_score = 1
        except Exception as e:
            #Verify that Java can continue from here
            #Maybe safely restart Java environment
            print(e)
            self.restart()
            compile_score = 0
        except SystemExit:
            print("SystemExit")
            self.restart()
            compile_score = 0
        return compile_score
    
    def targets(self, agent, lvls, path=""):
        play_length = []
        win_loss = []
        compiles = []
        for i, lvl in enumerate(lvls):
            name = os.path.join(path, 'lvl_{}.txt'.format(i))
            compiled = self.load_level(lvl, name)
            if(compiled > 0):
                score, steps, win = agent.play(self.env, 1)
            else:
                score, steps, win = 0, 0, 0
            compiles.append(compiled)
            play_length.append(steps)
            win_loss.append(win)
        return compiles, win_loss, play_length
    
    def record(self, lvl, compiled, win, length, file):
        add_header = not os.path.exists(file)   
        with open(file, 'a+') as results:
            writer = csv.writer(results)
            if(add_header):
                writer.writerow(['lvl', 'compiles', 'win', 'length'])
            if(type(lvl) == list):
                for row in zip(lvl, compiled, win, length):
                    writer.writerow(row)
            else:
                writer.writerow((lvl, compiled, win, length))
     
    @timeout_decorator.timeout(1)
    def set_level(self, name):
        self.env._setLevel(name)
        self.env.reset()
        self.env.step(0)
        self.env.reset()
    
    def restart(self):
        #self.env.GVGAI.java.kill()
        self.env = gym_gvgai.make('gvgai-{}-lvl0-v0'.format(self.name)).unwrapped