"""
Wrappers for VGDL Games
"""
import random
import numpy as np

import gym
import gym_gvgai

#Look at baseline wrappers and make a wrapper file: New vec_wrapper + game_wrapper
class GridGame(gym.Wrapper):
    def __init__(self, game, play_length, shape, lvl_map=None, generator=None):
        """Returns Grid instead of pixels
        Also unifies the reward
        Step reward: -1/play_length
        Win reward: 1
        Lose reward: -1
        Total reward = 1 - steps/play_length
        #PPO wants to maximize, Generator wants a score of 0
        --------
        """
        gym.Wrapper.__init__(self, env)
        self.name = game
        self.env = gym_gvgai.make('gvgai-{}-lvl0-v0'.format(game))
        self.levels = level.Levels(generator, lvl_map) if generator==None else None

        self.compiles = True
        self.state = None
        self.play_length = play_length
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float)

    def reset(self):
    	state = self.set_level()
    	return state

    def step(self, action):
        if(not self.compiles):
           return self.state, -10, True, {}
        _, _, done, info = self.env.step(action)
        reward = self.get_reward(done, info["winner"] )
        state = self.get_state(info['grid'])
        return state, reward, done, {}

    def get_reward(self, isOver, winner):
        if(isOver):
            if(winner == 'PLAYER_WINS'):
                return 1
            else:
                return -1
        else:
            return -1/self.play_length

    def get_state(self, grid):
        state = self.pad(grid)
        state = self.background(state)
        self.state = state
        return state

    def set_level(self):
        if(self.levels):
            state = self.levels.new_level()
            try:
                self.env.unwrapped._setLevel(self.levels.path)
                self.levels.test(self.env)
                self.compiles = True
            except Exception as e:
                #print(e)
                self.compiles = False
                self.restart()
            except SystemExit:
                #print("SystemExit")
                self.compiles = False
                self.restart()
        else:
            lvl = random.randint(0,4)
            self.env.unwrapped._setLevel(lvl)
            _, _, _, info = self.env.step(0)
            state = self.get_state(info['grid'])
        self.state = state
        return state
        
    def pad(self, state):
        pad_h = max(self.shape[-2] - state.shape[-2], 0)
        pad_w = max(self.shape[-1] - state.shape[-1], 0)
        pad_height = (pad_h//2, pad_h-pad_h//2)
        pad_width = (pad_w//2, pad_w-pad_w//2)
        padding = ((0,0), pad_height, pad_width)
        return np.pad(state, padding, 'constant')
        
    def background(self, state):
        background = 1 - np.clip(np.sum(state, 0), 0, 1)
        background = np.expand_dims(background, 0)
        return np.concatenate([state, background])

    def restart(self):
        self.env = gym_gvgai.make('gvgai-{}-lvl0-v0'.format(self.name))