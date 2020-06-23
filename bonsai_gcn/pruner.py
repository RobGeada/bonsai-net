import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from torch.autograd import Variable

def inverse_sig(m, x):
    return np.log(x/(1-x))/m

class Pruner(nn.Module):
    def __init__(self, m=1e5, mem_size=0, init=None):
        super().__init__()
        self.mem_size = mem_size
        self.weight = nn.Parameter(torch.tensor([init]))
        self.m = m
        #self.gate = lambda w: (.5 * w / torch.abs(w)) + .5
        self.gate = lambda w: w>0
        self.saw = lambda w: (self.m * w - torch.floor(self.m * w)) / self.m
        self.deadheaded = False
        self.weight_history = []
        self.actual_weights = []

    def __str__(self):
        return 'Pruner'

    def num_params(self):
        # return number of differential parameters of input model
        return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    def track_gates(self, deadhead, print_weights):
        self.weight_history.append(self.gate(self.weight).item())
        self.actual_weights.append(self.weight.item())
        return self.deadhead(deadhead, print_weights=print_weights)
        
        #self.weight_history.append(self.weight.item()>self.thresh)
        #self.actual_weights.append(self.weight.item())

    def get_deadhead(self, deadhead_epochs, print_weights, verbose=False):
        if len(self.weight_history)<deadhead_epochs:
            return False
        deadhead = not any(self.weight_history[-deadhead_epochs:])
        if 0:#print_weights:
            print(np.array(self.actual_weights)[-deadhead_epochs:])
        if verbose:
            print(self.weight_history, deadhead)
        #self.weight_history = []
        return deadhead

    def deadhead(self, deadhead, print_weights):
        if not self.deadheaded and self.get_deadhead(deadhead, print_weights=print_weights):
            self.deadheaded=True
            self.weight_history = []
            for param in self.parameters():
                param.requires_grad = False
            return 1
        else:
            return 0

    def sg(self):
        return self.saw(self.weight) + self.gate(self.weight)

    def forward(self, x):
        return self.sg() * x
