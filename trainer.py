import os
import csv
import pathlib

import torch
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, gen, adapter, game, agent, save, version=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = gen.to(self.device)
        self.adapter = adapter.to(self.device)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr = 0.001) #0.0001
        self.game = game
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
    
    def save_models(self, version, g_loss, c_loss):
        self.agent.save(self.save_paths['agent'], version)
        torch.save({
            'generator_model': self.generator.state_dict(),
            'generator_optimizer': self.gen_optimizer.state_dict(),
            'version': version,
            'gen_loss': g_loss,
            'critic_loss': c_loss
            }, os.path.join(self.save_paths['models'], "checkpoint_{}.tar".format(version)))
    
    def save_loss(self, update, gen_loss, pred_loss, compile_loss, win_loss, length_loss):
        add_header = not os.path.exists(self.save_paths['loss'])   
        with open(self.save_paths['loss'], 'a+') as results:
            writer = csv.writer(results)
            if(add_header):
                header = ['update', 'gen_loss', 'pred_loss', 'compile_loss', 'win_loss', 'length_loss']
                writer.writerow(header)
            writer.writerow((update, gen_loss, pred_loss, compile_loss, win_loss, length_loss))
        
    def save_levels(self, tensor, compiled, win, length):
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
    
    def eval_levels(self, tensor):
        levels = self.game.create_levels(tensor)
        compiled, win, length = self.game.targets(self.agent, levels)
        c = torch.Tensor(compiled).unsqueeze(1).to(self.device)
        w = torch.Tensor(win).unsqueeze(1).to(self.device)
        l = torch.Tensor(length).unsqueeze(1).to(self.device)
        return c, w, l
        
    def train(self, updates, batch_size): #critic_updates
        self.generator.train()
        self.critic.train()
        z = self.z_generator(batch_size, self.generator.z_size)
        loss = np.zeros(5)
        
        #Pretrain on existing levels -> maybe a function in agent
        
        for update in range(self.version + 1, self.version + updates + 1):
#             self.unfreeze_weights(self.critic)
            
#             for c_update in range(critic_updates):
#                 levels = self.generator(z())
#                 compiled_real, win_real, length_real = self.eval_levels(levels)
                
#                 self.critic_optimizer.zero_grad()
#                 compiled_pred, win_pred, length_pred = self.critic(levels)
#                 c_loss = F.binary_cross_entropy(compiled_pred, compiled_real)
#                 w_loss = F.binary_cross_entropy(win_pred, win_real)
#                 l_loss = F.mse_loss(length_pred, length_real)
#                 prediction_loss = c_loss + w_loss + l_loss
#                 prediction_loss.backward()
#                 self.critic_optimizer.step()
            
            #NEW!!!! Don't need to eval anymore!!!!!!!!! what does actor critic loss expect?
            levels = self.generator(z())
            compiled_real, win_real, length_real = self.eval_levels(levels)
    
    
            #Save a generated level
            #if(update%10 == 0):
            self.save_levels(levels, compiled_real, win_real, length_real)
                
#            self.freeze_weights(self.critic)
            self.gen_optimizer.zero_grad()
            levels = self.generator(z())
            target = torch.ones(batch_size).to(self.device)
            compiled_pred, win_pred, length_pred = self.critic(levels)
            gen_loss = F.binary_cross_entropy(compiled_pred, target)
            gen_loss += F.binary_cross_entropy(win_pred, target)
            gen_loss += F.mse_loss(length_pred, target) #self.agent.play_length*target
            gen_loss.backward()
            self.gen_optimizer.step()

            loss += gen_loss.item(), prediction_loss.item(), c_loss.item(), w_loss.item(), l_loss.item()
            save_frequency = 10
            if(update%save_frequency == 0):
                self.save_models(update, gen_loss, prediction_loss)
                self.save_loss(update, *loss/save_frequency)
                out = '[{}] Gen Loss: {}; Pred Loss: {} -> C {}, W {}, S {}'
                out = out.format(update, *loss/save_frequency)
                loss = np.zeros(5)
                print(out)