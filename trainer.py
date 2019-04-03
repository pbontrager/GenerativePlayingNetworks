import os
import csv
import pathlib
import tempfile
import numpy as np

import torch
import torch.nn.functional as F

import pdb

class Trainer(object):
    def __init__(self, gen, agent, save, version=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = gen.to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr = 0.001) #0.0001
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

        if(version > 0):
            self.load(version)
        else:
            self.version = 0

    def load(self, version):
        self.version = version
        self.agent.load(self.save_paths['agents'], version)

        path = os.path.join(self.save_paths['models'], "checkpoint_{}.tar".format(version))
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_model'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])

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

    def new_levels(self, num):
        lvl_tensor, states = self.generator.new(num)
        lvl_strs = self.agent.env_def.create_levels(lvl_tensor)
        for i in range(num):
            path = os.path.join(self.temp_dir.name, "lvl_{}".format(i))
            np.save(path + ".npy", states[i].cpu().numpy())
            with open(path + ".txt", 'w') as file:
                file.write(lvl_strs[i])
        return lvl_strs, states
            
    def freeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def z_generator(self, batch_size, z_size):
        return lambda:torch.Tensor(batch_size, z_size).normal_().to(self.device)

    def critic(self, x):
        rnn_hxs = torch.zeros(x.size(0), self.agent.actor_critic.recurrent_hidden_state_size).to(self.device)
        masks = torch.ones(x.size(0), 1).to(self.device)
        return self.agent.actor_critic.get_value(x, rnn_hxs, masks)

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
        z = self.z_generator(batch_size, self.generator.z_size)

        loss = 0
        for update in range(self.version + 1, self.version + updates + 1):
            if(self.version == 0):
                self.agent.set_envs() #Pretrain on existing levels
            else:
                self.new_levels(batch_size)
                self.agent.set_envs(self.temp_dir.name)

            self.unfreeze_weights(self.agent.actor_critic.base)
            self.agent.train_agent(rl_steps)
            self.freeze_weights(self.agent.actor_critic.base)

            self.gen_optimizer.zero_grad()
            levels = self.generator(z())
            states = self.generator.adapter(levels)
            expected_value = self.critic(states)
            target = torch.ones(batch_size).to(self.device)
            gen_loss = F.mse_loss(expected_value, target)
            gen_loss.backward()
            self.gen_optimizer.step()

            #Save a generated level
            levels, states = self.new_levels(1)
            with torch.no_grad():
                expected_rewards = self.critic(states)
            #real_rewards = self.eval_levels(levels)
            real_rewards = 'Nan'
            self.save_levels(update, levels, real_rewards, expected_rewards)

            #Save and report results
            loss += gen_loss.item()
            self.version += 1
            save_frequency = 10
            if(update%save_frequency == 0):
                self.save_models(update, gen_loss)
                self.save_loss(update, loss/save_frequency)
                print('[{}] Gen Loss: {}'.format(update, loss/save_frequency))
                loss = 0
        self.agent.envs.close()
