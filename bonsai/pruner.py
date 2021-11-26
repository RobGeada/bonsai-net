import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable

from bonsai.ops import *


# === BASE PRUNER =================================================================================
class Pruner(nn.Module):
    def __init__(self, m=1e5, mem_size=0, init=None):
        super().__init__()
        if init is None:
            init = .01
        elif init is 'off':
            init = -1.
        elif type(init) is int:
            init = float(init)
        self.init = init
        self.mem_size = mem_size
        self.weight = nn.Parameter(torch.tensor([self.init]))
        self.m = torch.tensor(m)
        self.m_inv = 1/m
        
        self.weight_history = []

    def __str__(self):
        return 'Pruner: M={},N={}'.format(self.M, self.channels)

    def num_params(self):
        # return number of differential parameters of input model
        return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    def track_gates(self):
        #self.actual_weight_vals.append(self.weight.item())
        self.weight_history.append(self.gate().item())

    def get_deadhead(self, prune_interval, verbose=False):
        if len(self.weight_history)<prune_interval:
            return False
        deadhead = (prune_interval * .25) > sum(self.weight_history[-prune_interval:])
        if deadhead:
            self.switch_off()
        self.weight_history = self.weight_history[-prune_interval:]
        
        if verbose:
            print(self.weight_history, deadhead)
        return deadhead

    def clamp(self):
        pre = self.weight.item()
        bound = self.init * 5
        if self.weight > bound:
            self.weight.data = self.weight.data * bound/self.weight.data
        elif self.weight < -bound:
            self.weight.data = self.weight.data * -bound/self.weight.data
    
    def switch_off(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def saw(self):
        return torch.remainder(self.weight, self.m_inv)
    
    def gate(self):
        return self.weight > 0

    def sg(self):
        return self.saw() + self.gate()
          

    def forward(self, x):
        return self.sg() * x

    
        
    

# === OP + PRUNER COMBO =================================================================================
class PrunableOperation(nn.Module):
    def __init__(self, op_function, name, mem_size, c_in, stride, pruner_init=None, prune=True):
        super().__init__()
        self.op_function = op_function
        self.stride = stride
        self.name = name
        self.op = self.op_function(c_in, stride)
        self.zero = name == 'Zero'
        self.prune = prune
        if self.prune:
            self.pruner = Pruner(mem_size=mem_size, init=pruner_init)
        if pruner_init is 'off':
            self.zero = True
            self.pruner.switch_off()

    def track_gates(self):
        self.pruner.track_gates()

    def deadhead(self,prune_interval):
        if self.zero or not self.pruner.get_deadhead(prune_interval):
            return 0
        else:
            self.op = Zero(self.stride)
            self.zero = True
            return 1

    def __str__(self):
        return self.name

    def forward(self, x, fw_type=None):
        if self.prune:
            out = self.op(x) if self.zero else self.pruner(self.op(x))
                                                           
        else:
            out = self.op(x)
        return out
    


# === INPUT HANDLER FOR PRUNED CELL INPUTS
class PrunableInputs(nn.Module):
    def __init__(self, dims, scale_mod, genotype, random_ops, prune=True):
        super().__init__()
        # weight inits
        if genotype.get('weights') is None:
            pruner_inits = [None]*len(dims)
        else:
            pruner_inits = genotype.get('weights')

        # zero inits
        if genotype.get('zeros') is None:
            self.zeros = [False] * len(dims)
        else:
            self.zeros = genotype.get('zeros')

        if random_ops is not None:
            self.zeros = [False if np.random.rand()<(random_ops['i_c']/len(self.zeros)) else True for i in self.zeros]
            if all(self.zeros):
                self.zeros[np.random.choice(range(len(self.zeros)))]=False
        self.unified_dim = dims[-1]
        self.prune = prune
        ops, strides, upscales, pruners = [], [], [], []

        for i, dim in enumerate(dims):
            stride = self.unified_dim[1]//dim[1] if dim[3] != self.unified_dim[3] else 1
            strides.append(stride)
            c_in, c_out = dim[1], self.unified_dim[1]
            upscales.append(c_out/c_in)
            if self.zeros[i]:
                ops.append(Zero(stride, c_out/c_in))
            else:
                ops.append(MinimumIdentity(c_in, c_out, stride))
            if self.prune:
                pruners.append(Pruner(init=pruner_inits[i]))

        self.ops = nn.ModuleList(ops)
        self.strides = strides
        self.upscales = upscales
        if self.prune:
            self.pruners = nn.ModuleList(pruners)
        self.scaler = MinimumIdentity(self.unified_dim[1], self.unified_dim[1]*scale_mod, stride=1)

    def track_gates(self):
        [pruner.track_gates() for pruner in self.pruners]

    def deadhead(self, prune_interval):
        out = 0
        for i, pruner in enumerate(self.pruners):
            if self.zeros[i] or not pruner.get_deadhead(prune_interval):
                out += 0
            else:
                self.ops[i] = Zero(self.strides[i], self.upscales[i])
                self.zeros[i] = True
                out += 1
        return out

    def get_ins(self):
        return [i - 1 if i else 'In' for i, zero in enumerate(self.zeros) if not zero]

    def __str__(self):
        return str(self.get_ins())

    def forward(self, xs, fw_type=None):
        out = None
        if self.prune:
            for i, op in enumerate(self.ops):
                if out is None:
                    out = op(xs[i]) if self.zeros[i] else self.pruners[i](op(xs[i]))
                else:
                    out = out + op(xs[i]) if self.zeros[i] else self.pruners[i](op(xs[i]))
        else:
            for op in self.ops:
                if out is None:
                    out = op(xs[i])
                else:
                    out = out + op(xs[i])
        return self.scaler(out)