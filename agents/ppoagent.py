import copy
import glob
import os
import time
import pathlib
import csv
import atexit
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.storage import RolloutStorage
#from a2c_ppo_acktr.evaluation import evaluate

from game.wrappers import make_vec_envs
from models.policy import Policy

#import pdb; pdb.set_trace()
import pdb
#Use inheritance to add a2c and acktr agents
class PPOAgent:
    #algorithm
    algo = 'ppo' #a2c, ppo, acktr
    use_gae = True #generalized advantage estimation
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5 #max norm of gradients
    clip_param = 0.1 #ppo clip

    seed = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cuda_deterministic = False
    no_cuda = False
    use_proper_time_limits = False
    use_linear_lr_decay = True

    #experiment setup
    num_steps = 128 #steps before an update
    ppo_epoch = 4
    num_mini_batch = 4
    log_interval = 1 #log per n updates
    save_dir = './trained_models/'
    log_dir = os.path.expanduser('/tmp/gym')
    eval_log_dir = log_dir + "_eval"
    save_interval = 100
    eval_interval = None
    recurrent_policy = True

    #optimization, RMSprop
    eps = 1e-5 #epsilon
    alpha = 0.99
    gamma = 0.99

    #imitation learning with gail
    gail_batch_size = 128
    gail_epoch = 5

    def __init__(self, env_def, generator, processes=1, version=0, lr=2.5e-4):
        self.env_def = env_def
        self.gen = generator
        self.num_processes = processes #cpu processes
        self.lr = lr
        self.version = version

        #Setup
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        if(self.num_mini_batch > processes):
            self.num_mini_batch = processes

        #State
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if not self.no_cuda and torch.cuda.is_available() and self.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        utils.cleanup_log_dir(self.log_dir)
        utils.cleanup_log_dir(self.eval_log_dir)

        torch.set_num_threads(1)

        self.handmade = None
        self.envs = None
        self.set_handmade_envs()

        if(version > 0):
            self.actor_critic = self.load(path, version)
        else:
            self.actor_critic = Policy(
                self.envs.observation_space.shape,
                self.envs.action_space,
                base_kwargs={'recurrent': self.recurrent_policy,
                             'shapes': list(reversed(self.env_def.model_shape))})
        self.actor_critic.to(self.device)

        if self.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                self.value_loss_coef,
                self.entropy_coef,
                lr=self.lr,
                eps=self.eps,
                alpha=self.alpha,
                max_grad_norm=self.max_grad_norm)
        elif self.algo == 'ppo':
            self.agent = algo.PPO(
                self.actor_critic,
                self.clip_param,
                self.ppo_epoch,
                self.num_mini_batch,
                self.value_loss_coef,
                self.entropy_coef,
                lr=self.lr,
                eps=self.eps,
                max_grad_norm=self.max_grad_norm)
        elif self.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, self.value_loss_coef, self.entropy_coef, acktr=True)

        self.gail = False
        self.gail_experts_dir = './gail_experts'
        if self.gail:
            assert len(self.envs.observation_space.shape) == 1
            self.gail_discr = gail.Discriminator(
                self.envs.observation_space.shape[0] + self.envs.action_space.shape[0], 100,
                self.device)
            file_name = os.path.join(
                self.gail_experts_dir, "trajs_{}.pt".format(
                    env_name.split('-')[0].lower()))

            self.gail_train_loader = torch.utils.data.DataLoader(
                gail.ExpertDataset(
                    file_name, num_trajectories=4, subsample_frequency=20),
                batch_size=self.gail_batch_size,
                shuffle=True,
                drop_last=True)

        self.rollouts = RolloutStorage(self.num_steps, self.num_processes,
                                  self.envs.observation_space.shape, self.envs.action_space,
                                  self.actor_critic.recurrent_hidden_state_size)

    def load(self, path, version):
        policy, ob_rms = torch.load(os.path.join(path, "agent_{}.tar".format(version)))
        print("Not using ob_rms: {}".format(ob_rms))
        utils.get_vec_normalize(self.envs).ob_rms = ob_rms
        self.actor_critic = policy

    def save(self, path, version):
        ob_rms = getattr(utils.get_vec_normalize(self.envs), 'ob_rms', None)
        torch.save([self.actor_critic, ob_rms], 
            os.path.join(path, "agent_{}.tar".format(version)))

    def report(self, version, total_num_steps, FPS, rewards):
        file_path = os.path.join(self.save_dir, "actor_critic_results.csv")
        add_header = not os.path.exists(file_path)
        if(len(rewards) > 0):
            mean, median, min, max = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
        else:
            mean, median, min, max = np.nan, np.nan, np.nan, np.nan
        with open(file_path, 'a+') as results:
            writer = csv.writer(results)
            if(add_header):
                header = ['update', 'total_steps', 'FPS', 'mean_reward', 'median_reward', 'min_reward', 'max_reward']
                writer.writerow(header)
            writer.writerow((version, total_num_steps, FPS, mean, median, min, max))

    def set_handmade_envs(self):
        if(not self.handmade):
            if(self.envs is not None):
                self.envs.close()
            self.envs = make_vec_envs(self.env_def, None, self.seed, self.num_processes, self.gamma, self.log_dir, self.device, True)
            self.handmade = True

    def set_generated_envs(self):
        if(self.handmade):
            if(self.envs is not None):
                self.envs.close()
            self.envs = make_vec_envs(self.env_def, self.gen, self.seed, self.num_processes, self.gamma, self.log_dir, self.device, True)
            self.handmade = False

    def play(self, env, runs=1, visual=False):
        score_mean, step_mean, win_mean = 0, 0, 0
        for i in range(runs):
            score, steps, win = self.play_game(env, visual)
            score_mean += score/runs
            step_mean += steps/runs
            win_mean += win/runs
        return score_mean, step_mean, win_mean

    def play_game(self, env, visual=False):
        raise Exception("Not implemented")

    def train_agent(self, num_env_steps):
        env_name = self.env_def.name

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(num_env_steps) // self.num_steps // self.num_processes
        for j in range(num_updates):

            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    self.agent.optimizer, j, num_updates,
                    self.agent.optimizer.lr if self.algo == "acktr" else self.lr)

            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = self.envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            if self.gail:
                if j >= 10:
                    self.envs.venv.eval()

                gail_epoch = self.gail_epoch
                if j < 10:
                    gail_epoch = 100  # Warm up
                for _ in range(gail_epoch):
                    self.gail_discr.update(self.gail_train_loader, self.rollouts,
                                 utils.get_vec_normalize(self.envs)._obfilt)

                for step in range(self.num_steps):
                    self.rollouts.rewards[step] = self.gail_discr.predict_reward(
                        self.rollouts.obs[step], self.rollouts.actions[step], self.gamma,
                        self.rollouts.masks[step])

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma,
                                     self.gae_lambda, self.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

            self.rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            total_num_steps = (j + 1) * self.num_processes * self.num_steps
            end = time.time()
            if (j % self.save_interval == 0 or j == num_updates - 1) and self.save_dir != "":
                self.version += 1
                #self.save(self.version)
                self.report(self.version, total_num_steps, int(total_num_steps / (end - start)), episode_rewards)

            if j % self.log_interval == 0 and len(episode_rewards) > 1:
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

            if (self.eval_interval is not None and len(episode_rewards) > 1
                    and j % self.eval_interval == 0):
                raise Exception("Evaluate not implemented yet")
                ob_rms = utils.get_vec_normalize(self.envs).ob_rms
                evaluate(self.actor_critic, ob_rms, env_name, self.seed,
                         self.num_processes, self.eval_log_dir, self.device)
