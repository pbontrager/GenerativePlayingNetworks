import os
import csv
import pathlib
import tempfile
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
import level_visualizer

import pdb


class Trainer(object):
    def __init__(self, gen, agent, save, version=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = gen.to(self.device)
        self.gen_optimizer = gen.optimizer #torch.optim.Adam(self.generator.parameters(), lr = 0.0001) #0.0001
        self.agent = agent
        self.temp_dir = tempfile.TemporaryDirectory()

        self.save_paths = {'dir':save}
        self.save_paths['agent'] = os.path.join(save,'agents')
        self.save_paths['models'] = os.path.join(save,'models')
        self.save_paths['levels'] = os.path.join(save,'levels.csv')
        self.save_paths['loss'] = os.path.join(save,'losses.csv')

        #Ensure directories exist
        pathlib.Path(self.save_paths['agent']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.save_paths['models']).mkdir(parents=True, exist_ok=True)

        self.level_visualizer = level_visualizer.LevelVisualizer(self.agent.env_def.name)

        if(version > 0):
            self.load(version)
        else:
            self.version = 0

    def load(self, version):
        self.version = version
        self.agent.load(self.save_paths['agent'], version)

        path = os.path.join(self.save_paths['models'], "checkpoint_{}.tar".format(version))
        if(os.path.isfile(path)):
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['generator_model'])
            self.gen_optimizer.load_state_dict(checkpoint['generator_optimizer'])

    def save_models(self, version, g_loss):
        self.agent.save(self.save_paths['agent'], version)
        torch.save({
            'generator_model': self.generator.state_dict(),
            'generator_optimizer': self.gen_optimizer.state_dict(),
            'version': version,
            'gen_loss': g_loss,
            }, os.path.join(self.save_paths['models'], "checkpoint_{}.tar".format(version)))

    def save_loss(self, update, gen_loss):
        add_header = not os.path.exists(self.save_paths['loss'])
        with open(self.save_paths['loss'], 'a+') as results:
            writer = csv.writer(results)
            if(add_header):
                header = ['update', 'gen_loss']
                writer.writerow(header)
            writer.writerow((update, gen_loss))

    def save_levels(self, update, strings, rewards, expected_rewards):
        add_header = not os.path.exists(self.save_paths['levels'])
        with open(self.save_paths['levels'], 'a+') as results:
            writer = csv.writer(results)
            if(add_header):
                header = ['update', 'level', 'reward', 'expected_reward']
                writer.writerow(header)
            for i in range(len(strings)):
                writer.writerow((update, strings[i], rewards[i], expected_rewards[i].item()))

    def new_elite_levels(self, z):
        num = z.size(0)
        rewards = []
        elite_lvls = []
        no_compile = 0
        #debug
        #old_elites = []
        #debug_nc = set([int(file[4:-11]) for file in os.listdir(self.temp_dir.name) if file.endswith('.no_compile')])
        for file in os.listdir(self.temp_dir.name):
            path = os.path.join(self.temp_dir.name, file)
            #debug
            #if(file is 'rewards.csv'):
            #    data = np.genfromtxt(path, delimiter=',', skip_header=1)
            #    old_elites = data[:, 0].astype('int').tolist()
            if(file.endswith('.csv')):
                data = np.genfromtxt(path, delimiter=',', skip_header=1)
                if(data.ndim == 1):
                    data = np.expand_dims(data, 0)
                rewards.append(data)
                os.remove(path)
            elif(file.endswith('.no_compile')):
                no_compile += 1
                os.remove(path)
        if(len(rewards) > 0):
            rewards = np.concatenate(rewards)
            rewards = pd.DataFrame(rewards).groupby(0)
            avg_rewards = rewards.mean()
            rewards = rewards.max()
            #debug
            #debug_played = set(rewards.index.astype('int').tolist())
            #debug_played_nc = debug_played.intersection(debug_nc)
            #if(len(debug_played_nc) > 0):
            #    nc_stats = {}
            #    for nc in debug_played_nc:
            #        with open(self.temp_dir.name + '/lvl_{}.txt'.format(nc), 'r') as f:
            #            nc_lvl = f.read()
            #        nc_stats[nc] = [nc_lvl.find('A'), nc_lvl.find('g'), nc_lvl.find('+')]
            #    if(min([min(i) for i in nc_stats.values()]) < 0):
            #        pdb.set_trace()

            winning_rewards = rewards[rewards[1] > 0].sort_values(1)
            losing_rewards = rewards[rewards[1] <= 0].sort_values(1, ascending=False)
            rewards = pd.concat([winning_rewards, losing_rewards])
            elite_lvls = rewards.index.astype('int')[:num//3].tolist()
            if(len(elite_lvls) > 0):
                with open(os.path.join(self.temp_dir.name,'rewards.csv'), 'w') as logs:
                    writer = csv.writer(logs)
                    writer.writerow(['level','reward'])
                    for lvl in elite_lvls:
                        reward = rewards.loc[lvl].item()
                        writer.writerow([lvl, reward])
        #debug
        print(elite_lvls)
        rewards = np.mean(avg_rewards.values) if not type(rewards)==list else 0
        self.agent.writer.add_scalar('levels/Elite Levels', len(elite_lvls), self.version)
        self.agent.writer.add_scalar('levels/Level Reward', rewards, self.version)
        self.agent.writer.add_scalar('levels/Uncompilable Levels', no_compile, self.version)
        lvl_tensor, states = self.generator.new(z)
        lvl_strs = self.agent.env_def.create_levels(lvl_tensor)
        elite_images = []
        for i in range(num):
            path = os.path.join(self.temp_dir.name, "lvl_{}".format(i))
            if(i not in elite_lvls):
                np.save(path + ".npy", states[i].cpu().numpy())
                with open(path + ".txt", "w") as file:
                    file.write(lvl_strs[i])
                if(not self.agent.env_def.pass_requirements(lvl_strs[i])):
                    open(path + ".no_compile", "w").close()
            else:
                with open(path + ".txt") as f:
                    lvl_strs[i] = f.read()
                states[i] = torch.Tensor(np.load(path + ".npy")).to(self.device)
                elite_images.append(np.array(self.level_visualizer.draw_level(lvl_strs[i]))/255.0)
        if(len(elite_images) > 0):
            self.agent.writer.add_image('Elite Levels', make_grid(elite_images[:8], pad_value=1), (self.version), dataformats='HWC')
        return lvl_strs, states

    def new_levels(self, z, save=False):
        lvl_tensor, states = self.generator.new(z)
        lvl_strs = self.agent.env_def.create_levels(lvl_tensor)
        num = z.size(0) if save else 0
        for i in range(num):
            path = os.path.join(self.temp_dir.name, "lvl_{}".format(i))
            np.save(path + ".npy", states[i].cpu().numpy())
            with open(path + ".txt", 'w') as file:
                file.write(lvl_strs[i])
            if(not self.agent.env_def.pass_requirements(lvl_strs[i])):
                open(path + ".no_compile", "w").close()
        return lvl_strs, states

    def freeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def z_generator(self, batch_size, z_size):
        return lambda b=batch_size, z=z_size:torch.Tensor(b, z).normal_().to(self.device)

    def critic(self, x):
        self.agent.agent.optimizer.zero_grad()
        rnn_hxs = torch.zeros(x.size(0), self.agent.actor_critic.recurrent_hidden_state_size).to(self.device)
        masks = torch.ones(x.size(0), 1).to(self.device)
        actions = torch.zeros_like(masks).long()
        value, _, _, _, dist_entropy, _ = self.agent.actor_critic.evaluate_actions(x, rnn_hxs, masks, actions)
        return value, dist_entropy
        #return self.agent.actor_critic.get_value(x, rnn_hxs, masks)

    def eval_levels(self, tensor):
        #raise Exception("Not implemented")
        #levels = self.game.create_levels(tensor)
        #What to pass to play?
        #File Names?
        #Create new envs for evaluation...
        rewards = self.agent.play(levels)
        return rewards

    def train(self, updates, batch_size, rl_steps):
        self.generator.train()
        z = self.z_generator(batch_size, 128) #self.generator.z_size) scale debug

        #scale debug
        #scale = nn.Sequential(
        #    nn.Linear(64, 128, bias=False), nn.ReLU(),
        #    nn.Linear(128, 256, bias=False), nn.ReLU(),
        #    nn.Linear(256, 512, bias=False), nn.ReLU())
        scale = nn.Linear(128, 512)
        scale.to(self.device)
        scale_optim = torch.optim.Adam(scale.parameters(), lr=1e-4) #scale debug

        loss = 0
        entropy = 0
        gen_updates = 0
        for update in range(self.version + 1, self.version + updates + 1):
            if(self.version == -1): #debug
                self.agent.set_envs() #Pretrain on existing levels
                self.agent.train_agent(200*rl_steps) #200*rl_steps
                self.save_models(0, 0)
            elif(self.version >= 0): #debug was 1
                self.new_elite_levels(scale(z(128))) #batch_size) scale debug
                self.agent.set_envs(self.temp_dir.name)
                self.agent.train_agent(rl_steps)

            #Not updating, range = 0
            generated_levels = []
            for i in range(5):
                levels, _ = self.new_levels(scale(z(8)))
                lvl_imgs = [np.array(self.level_visualizer.draw_level(lvl))/255.0 for lvl in levels]
                generated_levels = lvl_imgs

                self.gen_optimizer.zero_grad()
                scale_optim.zero_grad() #scale debug
                levels = self.generator(scale(z()))
                states = self.generator.adapter(levels)
                expected_value, dist = self.critic(states)
                '''
                TODO:
                    Use add_histograms to show distribution of enemy, avatar, key, door, wall counts
                    Use network reconstruction for generating
                    Implement InfoGAN metrics

                    Limit catastrophic negative policy values
                    Treat human levels and good levels like replay buffer

                    Generator should push entropy up, while agent pushes it down
                    Level Elites - only take the last version of every level
                '''
                diversity = (states[:-1] - states[1:]).pow(2).mean()
                target = 1*torch.ones_like(expected_value) #Best
                #target = .5 + torch.rand_like(expected_value)
                #target_dist = torch.ones_like(dist)
                #gen_loss = F.binary_cross_entropy_with_logits(expected_value, target)
                gen_loss = F.mse_loss(expected_value, target)
                #gen_loss += -.01*dist #F.mse_loss(dist, target_dist)
                div_loss = -diversity
                loss = gen_loss + dist #+ div_loss #.1*dist
                loss.backward()
                self.gen_optimizer.step() #scale debug
                scale_optim.step()  #scale Debug

                self.agent.writer.add_scalar('generator/loss', gen_loss.item(), gen_updates)
                self.agent.writer.add_scalar('generator/entropy', dist.item(), gen_updates)
                self.agent.writer.add_scalar('generator/diversity', diversity.item(), gen_updates)

                gen_updates += 1
                if(gen_loss.item() < .1):
                    print("Updated gen {} times".format(i))
                    break

            self.agent.writer.add_image('Generated Levels', make_grid(generated_levels, pad_value=1), (update-1), dataformats='HWC')
            #Save a generated level
            levels, states = self.new_levels(scale(z(1))) #scale debug
            with torch.no_grad():
                expected_rewards = self.critic(states)
            #real_rewards = self.eval_levels(levels)
            real_rewards = ['Nan']
            self.save_levels(update, levels, real_rewards, expected_rewards)

            #Save and report results
            loss += gen_loss.item()
            entropy += dist.item()
            self.version += 1
            save_frequency = 100
            if(update%save_frequency == 0):
                self.save_models(update, gen_loss)
                self.save_loss(update, loss/save_frequency)
                print('[{}] Gen Loss: {}, Entropy {}'.format(update, loss/save_frequency, entropy/save_frequency))
                loss = 0
                entropy = 0
        self.agent.envs.close()
