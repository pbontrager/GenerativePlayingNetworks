import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math

def sigmoidDist(x, grade=10):
    return F.sigmoid(grade*x)*(1 - F.sigmoid(grade*x))

def gaussianDist(x, sigma=.02):
    return (1/(sigma*math.sqrt(2*math.pi)))*(-.5*(x/sigma).pow(2)).exp()

def fuzzyDist(x, a=.1, b=2):
    return 1/(1 + (x/a).abs().pow(2*b))

class SoftHist(nn.Module):
    def __init__(self, bins, dist):
        super(SoftHist, self).__init__()

        binwidth = bins[1] - bins[0]
        self.bins = nn.Parameter(bins.unsqueeze(1))
        self.dist = dist

        self.pdf = lambda h: h/(h.sum())

    def forward(self, x):
        diffs = x.squeeze() - self.bins
        distances = self.dist(diffs)
        hist = distances.sum(1)
        hist_norm = self.pdf(hist)
        return hist_norm

class NormalDivLoss(nn.Module):
    def __init__(self, dist=fuzzyDist):
        super(NormalDivLoss, self).__init__()

        bins = torch.arange(-10, 10, .2)
        binwidth = bins[1] - bins[0]
        self.hist = SoftHist(bins, dist)
        self.kl = nn.KLDivLoss(reduction='batchmean')

        self.target = nn.Parameter(binwidth*torch.distributions.normal.Normal(0, .3).log_prob(bins).exp().unsqueeze(1))

    def forward(self, x):
        hist = self.hist(x)
        hist_log = torch.log(hist).unsqueeze(1)
        return self.kl(hist_log, self.target)
