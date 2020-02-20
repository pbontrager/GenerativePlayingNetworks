import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import mul
import math

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

import models.utils as utils
import pdb

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, model='base'):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                if(model=='base'):
                    base = CNNBase
                elif(model=='resnet'):
                    base = CNNDeep
                else:
                    raise Exception('Model not implemented')
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if(value.size(-1) > 1):
            Q = value.gather(1, action)
            value = (dist.probs*value).sum(1).unsqueeze(1)
        else:
            Q = value

        action_prob = dist.probs.gather(1, action)
        action_log_prob = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, Q, action, action_prob, action_log_prob, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, actor_features, _ = self.base(inputs, rnn_hxs, masks)
        if(value.size(-1) > 1):
            dist = self.dist(actor_features)
            value = (dist.probs*value).sum(1).unsqueeze(1)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if(value.size(-1) > 1):
            Q = value.gather(1, action)
            value = (dist.probs*value).sum(1).unsqueeze(1)
        else:
            Q = value

        action_prob = dist.probs.gather(1, action)
        action_log_prob = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, Q, action_prob, action_log_prob, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class CNNBase(NNBase):
    def __init__(self, num_inputs, shapes=[], recurrent=False, hidden_size=512, dropout=0):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.main = nn.Sequential(
             nn.Conv2d(num_inputs, 128, 3, padding=(3,1), stride=1), nn.ReLU(),
             nn.Conv2d(128, 64, 3, padding=1, stride=2), nn.ReLU(),
             nn.Conv2d(64, 32, 3, padding=1, stride=2), nn.ReLU(), Flatten(), nn.Dropout(dropout),
             nn.Linear(32*4*4, hidden_size), nn.ReLU())

        self.critic_linear = nn.Linear(hidden_size, 1)
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class ResidualBlock(nn.Module):
    def __init__(self, inputs, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inputs, channels, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(channels, inputs, 3, padding=1, stride=1),
            nn.ReLU())

    def forward(self, x):
        return x + self.block(x)

class CNNDeep(NNBase):
    def __init__(self, num_inputs, shapes=[], recurrent=False, hidden_size=512, dropout=0):
        super(CNNDeep, self).__init__(recurrent, hidden_size, hidden_size)

        self.block1 = nn.Sequential(
             nn.Conv2d(num_inputs, 16, 3, padding=(3,1), stride=1), nn.ReLU(),
             #nn.MaxPool2d(3, padding=1, stride=1), nn.ReLU(),
             ResidualBlock(16, 16))

        self.block2 = nn.Sequential(
             nn.Conv2d(16, 32, 4, padding=1, stride=2), nn.ReLU(),
             #nn.MaxPool2d(3, padding=1, stride=2), nn.ReLU(),
             ResidualBlock(32, 32))

        self.block3 = nn.Sequential(
             nn.Conv2d(32, 32, 4, padding=1, stride=2), nn.ReLU(),
             #nn.MaxPool2d(3, padding=1, stride=2), nn.ReLU(),
             ResidualBlock(32, 32))

        #self.block4 = nn.Sequential(
        #    nn.Conv2d(32, 32, 3, padding=1, stride=1), nn.ReLU(),
        #    nn.MaxPool2d(3, padding=1, stride=2), nn.ReLU(),
        #    ResidualBlock(32, 32))

        self.flat = nn.Sequential(Flatten(),
             nn.Linear(32*4*4, hidden_size),
             nn.ReLU())

        self.critic_linear = nn.Linear(hidden_size, 6)
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        #x = self.block4(x) #large input
        x = self.flat(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

#class CNNBase(NNBase):
    #def __init__(self, num_inputs, shapes=[], recurrent=False, hidden_size=512):
    #    super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
    #
    #    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
    #                           constant_(x, 0), nn.init.calculate_gain('relu'))
    #
    #    self.blocks = nn.ModuleList()
    #    in_ch = num_inputs
    #    out_ch = 32
    #    halfway = math.ceil(len(shapes)/2)
    #    for i, s in enumerate(shapes):
    #        block = nn.Sequential(
    #            utils.Resize(s),
    #            init_(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
    #            nn.ReLU())
    #        in_ch = out_ch
    #        if(i+1 == halfway and len(shapes)%2==0):
    #            out_ch = in_ch
    #        elif(i+1 < halfway):
    #            out_ch = in_ch*2
    #        else:
    #            out_ch = in_ch//2
    #        self.blocks.append(block)
    #    block = nn.Sequential(Flatten(), init_(nn.Linear(in_ch * reduce(mul, shapes[-1]), hidden_size)), nn.ReLU())
    #    self.blocks.append(block)
    #
    #    # self.main = nn.Sequential(
    #    #     init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
    #    #     init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
    #    #     init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
    #    #     init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
    #
    #    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
    #                           constant_(x, 0))
    #
    #    self.critic_linear = init_(nn.Linear(hidden_size, 1))
    #
    #    self.train()

    #def forward(self, inputs, rnn_hxs, masks):
    #     x = inputs
    #     for b in self.blocks:
    #         x = b(x)
    #
    #     if self.is_recurrent:
    #         x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
    #
    #     return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
