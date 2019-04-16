import gym
import gym_gvgai
import numpy as np
from game.wrappers import GridGame

#env = gym.make('gvgai-aliens-lvl0-v1')
env = GridGame('aliens', 2000, (14, 12, 32))

env.reset()
for i in range(1000):
    action = np.random.randint(0,2,1)
    obs, reward, done, info = env.step(action)
    if(done):
        break
#print(info['winner'])
print(reward)

#timeout_decorator problems
