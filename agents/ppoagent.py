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
#from a2c_ppo_acktr.storage import RolloutStorage
from agents.storage import RolloutStorage
from agents.a2c import A2C_ACKTR
from agents.ppo import PPO

from game.wrappers import make_vec_envs, GridGame
from models.policy import Policy
from models.reconstruction import Decoder

from tensorboardX import SummaryWriter

import pdb
#Use inheritance to add a2c and acktr agents
class PPOAgent:
    #algorithm
    algo = 'a2c'          #a2c, ppo, acktr
    use_gae = False       #generalized advantage estimation
    gae_lambda = 0.95
    entropy_coef = 0.01   #weight maximizing action entropy loss
    value_loss_coef = 0.1 #.5 #weight value function loss
    max_grad_norm = 0.5   #max norm of gradients

    #ppo hyperparameters
    clip_param = 0.2    #ppo clip
    num_steps = 5       #steps before an update
    ppo_epoch = 4
    num_mini_batch = 32

    seed = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda_deterministic = False
    no_cuda = False
    use_proper_time_limits = False
    use_linear_lr_decay = False

    #experimnent setup
    log_interval = 1 #log per n updates
    log_dir = os.path.expanduser('/tmp/gym')
    eval_log_dir = log_dir + "_eval"
    save_interval = 100
    eval_interval = None
    recurrent_policy = True

    #optimization, RMSprop and TD
    eps = 1e-5    #epsilon
    alpha = 0.99
    gamma = 0.99  #discount factor

    #imitation learning with gail
    gail_batch_size = 128
    gail_epoch = 5

    def __init__(self, env_def, processes=1, dir='.', version=0, lr=2.5e-4, reconstruct=None):
        self.env_def = env_def
        self.num_processes = processes #cpu processes
        self.lr = lr
        self.version = version
        self.save_dir = dir + '/trained_models/'

        #Setup
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        if(self.num_mini_batch > processes):
            self.num_mini_batch = processes

        self.writer = SummaryWriter()
        self.total_steps = 0

        #State
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if not self.no_cuda and torch.cuda.is_available() and self.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        utils.cleanup_log_dir(self.log_dir)
        utils.cleanup_log_dir(self.eval_log_dir)

        torch.set_num_threads(1)

        self.level_path = None
        self.envs = None
        self.num_envs = -1
        self.set_envs(num_envs=1)

        if(version > 0):
            self.actor_critic = self.load(path, version)
        else:
            self.actor_critic = Policy(
                self.envs.observation_space.shape,
                self.envs.action_space,
                base_kwargs={'recurrent': self.recurrent_policy,
                             'shapes': list(reversed(self.env_def.model_shape))})
        self.actor_critic.to(self.device)

        #Reconstruction
        self.reconstruct = reconstruct is not None
        if(self.reconstruct):
            #layers = self.envs.observation_space.shape[0]
            #shapes = list(self.env_def.model_shape)
            #self.r_model = Decoder(layers, shapes=shapes).to(self.device)
            reconstruct.to(self.device)
            self.r_model = lambda x: reconstruct.adapter(reconstruct(x))
            #self.r_model = lambda x: reconstruct.adapter(reconstruct(x)).clamp(min=1e-6).log()
            self.r_loss = nn.L1Loss(reduction='sum') #nn.NLLLoss() #nn.MSELoss()
            self.r_optimizer = reconstruct.optimizer #optim.Adam(reconstruct.parameters(), lr = .0001)

        if self.algo == 'a2c':
            self.agent = A2C_ACKTR(
                self.actor_critic,
                self.value_loss_coef,
                self.entropy_coef,
                lr=self.lr,
                eps=self.eps,
                alpha=self.alpha,
                max_grad_norm=self.max_grad_norm)
        elif self.algo == 'ppo':
            self.agent = PPO(
                self.actor_critic,
                self.clip_param,
                self.ppo_epoch,
                self.num_mini_batch,
                self.value_loss_coef,
                self.entropy_coef,
                lr=self.lr,
                eps=self.eps,
                max_grad_norm=self.max_grad_norm,
                use_clipped_value_loss=False)
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
        #utils.get_vec_normalize(self.envs).ob_rms = ob_rms
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

    def set_envs(self, level_path=None, num_envs=None):
        num_envs = num_envs if num_envs else self.num_processes
        if(level_path != self.level_path or self.envs is None or num_envs != self.num_envs):
            if(self.envs is not None):
                self.envs.close()
            self.level_path = level_path
            self.envs = make_vec_envs(self.env_def, level_path, self.seed, num_envs, self.gamma, self.log_dir, self.device, True)
        self.num_envs = num_envs

    def update_reconstruction(self, rollouts):
        s, p, l, w, h = list(rollouts.obs.size())
        x = rollouts.obs.view(-1, l, w, h)
        hidden = rollouts.recurrent_hidden_states.view(s*p, -1)
        mask = rollouts.masks.view(s*p, -1)
        #y = x.argmax(1)
        y = x

        self.r_optimizer.zero_grad()
        self.agent.optimizer.zero_grad()
        _, predictions, _ = self.actor_critic.base(x, hidden, mask)
        reconstructions = self.r_model(predictions)
        loss = self.r_loss(reconstructions, y) #.05
        loss.backward()
        self.r_optimizer.step()
        self.agent.optimizer.step()
        return loss

    def update_reconstruct_next(self, rollouts):
        #Mask frames that are not relevant
        mask = rollouts.masks.unfold(0, 2, 1).min(-1)[0]
        mask = mask.view(-1)
        mask = torch.nonzero(mask).squeeze()

        #Image Pairs
        l, w, h = list(rollouts.obs.size())[2:]
        img_pairs = rollouts.obs.unfold(0, 2, 1) #128, 8, 14, 12, 16, 2
        img_pairs = img_pairs.view(-1, l, w, h, 2)
        img_pairs = img_pairs[mask]
        x = img_pairs[:, :, :, :, 0]
        y = img_pairs[:, :, :, :, 1]

        #Input hidden states
        hidden_size = rollouts.recurrent_hidden_states.size(2)
        hidden = rollouts.recurrent_hidden_states[:-1].view(-1, hidden_size) #129, 8, 512
        hidden = hidden[mask]

        #Update model
        self.r_optimizer.zero_grad()
        mask = torch.ones_like(mask).float().unsqueeze(1)
        _, predictions, _ = self.actor_critic.base(x, hidden, mask)
        reconstructions = self.r_model(predictions)
        loss = .05*self.r_loss(reconstructions, y)        #model -> x or x and a? x already contains action features
        loss.backward()
        self.r_optimizer.step()
        print(loss.item()) #add loss weight
        return loss

    def play(self, env, runs=1, visual=False):
        env = GridGame()
        reward_mean = 0
        for i in range(runs):
            score = self.play_game(env, visual)
            reward_mean += score/runs
        return score_mean

    def play_game(self, level):
        eval_envs = make_vec_envs(env_name, self.seed + self.num_processes, self.num_processes,
                              None, eval_log_dir, device, True)

        vec_norm = utils.get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

        eval_episode_rewards = []

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(self.num_processes, self.actor_critic.recurrent_hidden_state_size).to(self.device)
        eval_masks = torch.zeros(self.num_processes, 1).to(self.device)

        while len(eval_episode_rewards) < 10:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action)
            eval_masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float32).to(device)

            if(done):
                print("Done!")
        eval_envs.close()

    def train_agent(self, num_env_steps):
        env_name = self.env_def.name

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        n = 30
        episode_rewards = deque(maxlen=n)
        episode_values = deque(maxlen=n)
        episode_end_values = deque(maxlen=n)
        episode_end_probs = deque(maxlen=n)
        episode_lengths = deque(maxlen=n)
        first_steps = [True for i in range(self.num_processes)]

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
                    value, Q, action, action_prob, action_log_prob, recurrent_hidden_states = \
                        self.actor_critic.act(self.rollouts.obs[step],
                            self.rollouts.recurrent_hidden_states[step],
                            self.rollouts.masks[step])

                # Observe reward and next obs
                obs, reward, done, infos = self.envs.step(action)

                for i, step in enumerate(first_steps):
                    if step:
                        episode_values.append(value[i].item())
                    elif(done[i]):
                        episode_end_values.append(Q[i].item())
                        episode_end_probs.append(action_log_prob[i].item())
                first_steps = done

                for info in infos:
                    if 'episode' in info.keys():
                        r = info['episode']['r']
                        l = info['episode']['l']
                        episode_rewards.append(r)
                        episode_lengths.append(l)
                        if(r < -2):
                            print('Reward: {} and Length: {}'.format(r, l))
                            pdb.set_break()

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, action, action_prob,
                                action_log_prob, value, Q, reward, masks, bad_masks)

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
            if(self.reconstruct):
                recon_loss = self.update_reconstruction(self.rollouts)
                self.writer.add_scalar('generator/Reconstruction Loss', recon_loss.item(), self.total_steps)

            self.rollouts.after_update()

            #Tensorboard Reporting
            self.total_steps += self.num_processes*self.num_steps
            self.writer.add_scalar('value/Mean Reward', np.mean(episode_rewards), self.total_steps)
            self.writer.add_scalar('value/Episode Mean Length', np.mean(episode_lengths), self.total_steps)
            self.writer.add_scalar('policy/Action Loss', action_loss, self.total_steps)
            self.writer.add_scalar('value/Value Loss', value_loss, self.total_steps)
            self.writer.add_scalar('policy/Distribution Entropy', dist_entropy, self.total_steps)
            self.writer.add_scalar('value/Win Probability', np.mean(np.array(episode_rewards) > 0), self.total_steps)
            self.writer.add_scalar('value/Starting Value', np.mean(episode_values), self.total_steps)
            #self.writer.add_scalar('value/Ending Value', np.mean(episode_end_values), self.total_steps)
            self.writer.add_scalar('value/Log Probs', np.mean(episode_end_probs), self.total_steps)

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
