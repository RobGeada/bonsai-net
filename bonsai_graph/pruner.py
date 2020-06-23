import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable

from bonsai.ops import *

class Pruner(nn.Module):
    def __init__(self, m=1e5, mem_size=0, init=None):
        super().__init__()

        self.mem_size = mem_size
        self.weight = nn.Parameter(torch.tensor([0.]))
        self.m = m

        self.lr = .0001
        self.thresh = -9*self.lr
        self.gate = lambda w: torch.sigmoid(1/self.lr*w)
        #self.gate = lambda w: (.5 * w / torch.abs(w)) + .5
        self.saw = lambda w: (self.m * w - torch.floor(self.m * w)) / self.m
        self.deadheaded = False
        self.weight_history = []
        self.actual_weights = []

    def __str__(self):
        return 'Pruner'

    def num_params(self):
        # return number of differential parameters of input model
        return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    def track_gates(self):
        self.weight_history.append(self.weight.item())#>self.thresh)
        print(self.weight_history)
        #self.actual_weights.append(self.weight.item())

    def get_deadhead(self, verbose=False):
        deadhead = not any(self.weight_history)
        if verbose:
            print(self.weight_history, deadhead)
        self.weight_history = []
        return deadhead

    def deadhead(self):
        if not self.deadheaded and self.get_deadhead():
            self.deadheaded=True
            for param in self.parameters():
                param.requires_grad = False
            return 1
        else:
            return 0

    def sg(self):
        return self.saw(self.weight) + self.gate(self.weight)

    def forward(self, x):
        if self.deadheaded:
            return torch.zeros_like(x)
        else:
            return self.sg() * x
