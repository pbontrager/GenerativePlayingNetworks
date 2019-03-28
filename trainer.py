import os
import csv
import pathlib

import torch
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, gen, agent, save, version=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = gen.to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr = 0.001) #0.0001
        self.agent = agent
        
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
        
    def save_levels(self, tensor, compiled, win, length):
        raise Exception("Not implemented")
        levels = self.game.create_levels(tensor)
        for i in range(tensor.size(0)):
            self.game.record(levels[i], compiled[i].item(), win[i].item(), length[i].item(), self.save_paths['levels'])
        
    def freeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = False
            
    def unfreeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = True
        
    def z_generator(self, batch_size, z_size):
        return lambda:torch.Tensor(batch_size, z_size).normal_().to(self.device)

    def critic(self, batch_size):
        rnn_hxs = torch.zeros(self.agent.num_steps + 1, batch_size, self.agent.actor_critic.recurrent_hidden_state_size)
        masks = torch.ones(self.agent.num_steps + 1, batch_size, 1)
        return lambda x: self.agent.actor_critic.get_value(x, rnn_hxs, masks).to(self.device)
    
    # def eval_levels(self, tensor):
    #     raise Exception("Not implemented")
    #     levels = self.game.create_levels(tensor)
    #     compiled, win, length = self.game.targets(self.agent, levels)
    #     c = torch.Tensor(compiled).unsqueeze(1).to(self.device)
    #     w = torch.Tensor(win).unsqueeze(1).to(self.device)
    #     l = torch.Tensor(length).unsqueeze(1).to(self.device)
    #     return c, w, l
        
    def train(self, updates, batch_size, rl_steps):
        self.generator.train()

        z = self.z_generator(batch_size, self.generator.z_size)
        critic = self.critic(batch_size)
        
        loss = 0
        for update in range(self.version + 1, self.version + updates + 1):
            if(self.version == 0):
                self.agent.set_handmade_envs() #Pretrain on existing levels
            else:
                self.agent.set_generator_envs()

            self.unfreeze_weights(self.agent.actor_critic.base)
            self.agent.train_agent(rl_steps)
            self.freeze_weights(self.agent.actor_critic.base)

            self.gen_optimizer.zero_grad()
            levels = self.generator(z())
            states = self.generator.adapter(levels)
            expected_value = critic(states)
            target = torch.ones(batch_size).to(self.device)
            gen_loss = F.mse_loss(expected_value, target)
            gen_loss.backward()
            self.gen_optimizer.step()

            #Save a generated level
            #levels = self.generator(z())
            #self.eval_levels()
            #self.save_levels(levels, expected_rewards, real_rewards)

            #Save and report results
            loss += gen_loss.item()
            save_frequency = 10
            if(update%save_frequency == 0):
                self.save_models(update, gen_loss)
                self.save_loss(update, loss/save_frequency)
                print('[{}] Gen Loss: {}'.format(update, loss/save_frequency))
                loss = 0