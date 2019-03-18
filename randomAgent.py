import numpy as np

class RandomAgent:
    def __init__(self, play_length, shape):
        self.play_length = play_length
        self.shape = shape

    def load(self, path, version):
        pass
    
    def save(self, path, version):
        pass
    
    def play(self, env, runs=1, visual=False):
        score_mean, step_mean, win_mean = 0, 0, 0
        for i in range(runs):
            score, steps, win = self.play_game(env, visual)
            score_mean += score/runs
            step_mean += steps/runs
            win_mean += win/runs
        return score_mean, step_mean, win_mean
            
    def play_game(self, env, visual=False):
        env.reset()
        score, isOver = 0, False
        for step in range(self.play_length):
            state, reward, isOver, winner = self.step(env, env.action_space.sample())
            score += reward
            if(isOver):
                break
        length = (step + 1)/self.play_length
        return score, length, winner
    
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
    
    def step(self, env, action):
        _, reward, isOver, debug = env.step(action)
        state = self.pad(debug['grid'])
        state = self.background(state)
        winner = 1 if debug["winner"] == 'PLAYER_WINS' else 0
        return state, reward, isOver, winner