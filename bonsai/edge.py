import random
from torch.autograd import Variable

from bonsai.ops import *
from bonsai.helpers import *


class Edge(nn.Module):
    def __init__(self, dim, origin, target, op_sizes, stride=1, genotype=None, aim_size=None, prune=True):
        super().__init__()
        self.stride = stride
        if genotype is not None:
            available_ops = {k: commons[k] for (k, weight) in genotype}
            op_weights = [weight for (k, weight) in genotype]
        elif aim_size is not None:
            used, possible, available_ops = 0, 0, {}
            for k, v in sorted(op_sizes.items(), key=lambda x: x[1], reverse=True):
                if (used+v) <= aim_size:
                    available_ops[k]=commons[k]
                    used += v
                possible += v
            self.used = used
            self.possible = possible
            op_weights = [1.]*len(available_ops)
        else:
            available_ops = commons
            self.used = sum(op_sizes.values())
            self.possible = sum(op_sizes.values())
            op_weights = [None]*len(commons)

        self.ops = []
        for i, (key, op) in enumerate(available_ops.items()):
            prune_op = PrunableOperation(op_function=op,
                                         name=key,
                                         c_in=dim[1],
                                         mem_size=op_sizes[key],
                                         stride=stride,
                                         pruner_init=op_weights[i],
                                         prune=prune)
            self.ops.append(prune_op)
        self.ops = nn.ModuleList(self.ops)

        if prune:
            self.pruners = [op.pruner for op in self.ops]

        self.num_ops = len([op for op in self.ops if not op.zero])
        self.dim = dim
        self.origin = origin
        self.target = target

        if self.num_ops:
            self.normalizer = nn.BatchNorm2d(dim[1])
        else:
            self.normalizer = Zero(stride=self.stride)

        self.mask_gen = lambda dim, drop_prob: Variable(torch.cuda.FloatTensor(dim, 1, 1, 1).bernoulli(1-drop_prob))

    def deadhead(self):
        dhs = sum([op.deadhead() for i, op in enumerate(self.ops)])
        self.used = sum([op.pruner.mem_size for op in self.ops if not op.zero])
        self.num_ops -= dhs
        if self.num_ops == 0:
            self.normalizer = Zero(stride=self.stride)
        return dhs

    def genotype_compression(self, soft_ops=0, hard_ops=0):
        for op in self.ops:
            if not op.zero:
                hard_ops += op.pruner.mem_size
            soft_ops += op.pruner.sg() * op.pruner.mem_size
        return soft_ops, hard_ops

    def __str__(self):
        return "{}->{}: {}, \t({:,} params)".format(
            self.origin,
            self.target,
            [str(op) for op in self.ops if (not op.zero)],
            general_num_params(self))

    def __repr(self):
        return str(self)

    def set_half_mask(self):
        self.mask_gen = lambda dim, drop_prob: Variable(torch.cuda.HalfTensor(dim, 1, 1, 1).bernoulli(1 - drop_prob))

    def forward(self, x, drop_prob):
        if self.num_ops:
            summed = self.normalizer(sum([op(x) for op in self.ops]))
        else:
            # zero out input directly
            summed = self.normalizer(x)

        if random.random() < drop_prob:
            summed = summed.div(1-drop_prob)
            summed = summed.mul(self.mask_gen(summed.size(0), drop_prob))
        return summed