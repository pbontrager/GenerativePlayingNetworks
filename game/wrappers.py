
"""
Wrappers for VGDL Games
"""
import os
import csv
import random
import numpy as np

#import timeout_decorator

import gym
import gym_gvgai

import a2c_ppo_acktr.envs as torch_env

from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import pdb

def make_env(env_def, path, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        if(path):
            env = GridGame(env_def.name, env_def.length, env_def.state_shape, path, id=rank)
        else:
            env = GridGame(env_def.name, env_def.length, env_def.state_shape, id=rank)

        #env.seed(seed + rank)
        obs_shape = env.observation_space.shape

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)
        return env
    return _thunk

def make_vec_envs(env_def, level_path, seed, num_processes, gamma, log_dir, device, allow_early_resets, num_frame_stack=None):

    envs = [make_env(env_def, level_path, seed, i, log_dir, allow_early_resets) for i in range(num_processes)]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    #if len(envs.observation_space.shape) == 1:
    #    if gamma is None:
    #        envs = VecNormalize(envs, ret=False)
    #    else:
    #        envs = VecNormalize(envs, gamma=gamma)

    envs = torch_env.VecPyTorch(envs, device)

    #if num_frame_stack is not None:
    #    envs = torch_env.VecPyTorchFrameStack(envs, num_frame_stack, device)
    #elif len(envs.observation_space.shape) == 3:
    #    envs = torch_env.VecPyTorchFrameStack(envs, 4, device)

    return envs

#Look at baseline wrappers and make a wrapper file: New vec_wrapper + game_wrapper
class GridGame(gym.Wrapper):
    def __init__(self, game, play_length, shape, path=None, id=0):
        """Returns Grid instead of pixels
        Sets the reward
        Generates new level on reset
        #PPO wants to maximize, Generator wants a score of 0
        --------
        """
        self.id = id
        self.name = game
        self.levels = path
        self.level_id = -1
        self.version = 1
        self.env = gym_gvgai.make('gvgai-{}-lvl0-v{}'.format(game, self.version))
        gym.Wrapper.__init__(self, self.env)

        self.compiles = False
        self.state = None
        self.steps = 0
        self.score = 0
        self.play_length = play_length
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

    def reset(self):
        self.steps = 0
        self.score = 0
        state = self.set_level()
        return state

    def step(self, action):
        action = action.item()
        if(not self.compiles):
            return self.state, -2.0, True, {}
        _, r, done, info = self.env.step(action)
        if(self.steps >= self.play_length):
            done = True
        reward = self.get_reward(done, info["winner"], r)
        state = self.get_state(info['grid'])
        self.steps += 1
        self.score += reward
        return state, reward, done, {}

    def get_reward(self, isOver, winner, r):
        if(isOver):
            if(winner == 'PLAYER_WINS'):
                reward = 1 #- self.steps/self.play_length
            else:
                reward = -1 #+ self.steps/self.play_length
            self.log_reward(self.score + reward)
            return reward
        else:
            #if(r > 0):
            #    return 1/self.play_length
            #else:
            return 0

    def get_state(self, grid):
        state = self.pad(grid)
        state = self.background(state)
        self.state = state.astype('float32')
        return state

    def set_level(self):
        if(self.levels and random.randint(1,32) > -1):
            level_names = [file for file in os.listdir(self.levels) if file.endswith('.txt')]
            selection = random.choice(level_names)[:-4]
            self.level_id = int(selection[4:])
            path = os.path.join(self.levels, selection)
            state = np.load(path + ".npy")
            if(os.path.isfile(path + ".no_compile")):
                self.compiles = False
            else:
                try:
                    self.env.unwrapped._setLevel(path + ".txt")
                    self.test_level()
                    self.compiles = True
                except Exception as e:
                    #print(e)
                    self.compiles = False
                    self.restart(e, path)
                except SystemExit:
                    #print("SystemExit")
                    self.compiles = False
                    self.restart("SystemExit", path)
        else:
            self.compiles = True
            self.level_id = -1             #So log_reward doesn't track the validity of this level
            lvl = random.randint(0,4)
            self.env.unwrapped._setLevel(lvl)
            self.env.reset()
            _, _, _, info = self.env.step(0)
            state = self.get_state(info['grid'])
        self.state = state
        return state

    def log(self, text):
        path = os.path.join(self.levels, "log_{}.txt".format(self.id))
        with open('log_{}.txt'.format(self.id), 'a+') as log:
            log.write(str(text) + "\n")

    def log_reward(self, reward):
        if(self.level_id >= 0):
            path = os.path.join(self.levels, 'rewards_{}.csv'.format(self.id))
            add_header = not  os.path.exists(path)
            with open(path, 'a+') as rewards:
                writer = csv.writer(rewards)
                if(add_header):
                    writer.writerow(['level', 'reward'])
                writer.writerow((self.level_id, reward))

    #@timeout_decorator.timeout(1, use_signals=False)
    def test_level(self):
        self.env.reset()
        self.env.step(0)
        self.env.reset()

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

    def restart(self, e, path):
        #self.log(e)
        open(path + ".no_compile", 'w').close()
        self.env = gym_gvgai.make('gvgai-{}-lvl0-v{}'.format(self.name, self.version))

class CenteredGym(gym.Wrapper):
    def __init__(self, env, mapping, ascii):
        self.env = env
        self.name = self.env.name
        gym.Wrapper.__init__(self, self.env)

        self.avatar = mapping[ascii.index('A')]

        d, h, w = self.env.shape
        self.shape = (d, h + 2*(h//2), w + 2*(w//2))
        self.pad = (self.shape[1]//2, self.shape[2]//2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.shape, dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        x, y = self.get_pos(obs)
        pad_dims = ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]))
        padded = np.pad(obs, pad_dims, mode='constant')
        centered  = padded[:, y:y + self.shape[1], x:x + self.shape[2]]
        return centered

    def get_pos(self, obs):
        #map = obs[self.avatar]
        #pos = np.unravel_index(map.argmax(), map.shape)
        y, x = np.argwhere(obs.argmax(0)==self.avatar)[0]
        return (x, y)
