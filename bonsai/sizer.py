import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../")))

import pickle as pkl
import pprint
import torch
import numpy as np
import torch.nn as nn

from bonsai.data_loaders import load_data
from bonsai.helpers import mem_stats, clean, sizeof_fmt
from bonsai.net import Net
from bonsai.ops import commons
from bonsai.pruner import PrunableOperation
from bonsai.trainers import size_test

import collections



if __name__ == '__main__':
    if len(sys.argv)> 1 and sys.argv[1]=='o':
        # get sizes of individual operations in network
        # read args
        clean(verbose=False)
        C = int(sys.argv[3])
        dim = [int(x) for x in sys.argv[2:-1]]
        stride = int(sys.argv[-1])

        # build op
        op_mems = {}
        input_tensor = torch.zeros(dim,requires_grad=True).cuda()
        trials = 5
        criterion =  nn.CrossEntropyLoss()
        comparison = torch.tensor([0], dtype=torch.long).cuda()
        for op, f in commons.items():
            sizes = []
            
            for i in range(trials):
                sm = mem_stats(False)
                sms = []
                op_f = PrunableOperation(f, op, mem_size=0, c_in=C, stride=stride).cuda()  
                for _ in range(trials):
                    sms.append(mem_stats(False)-sm)
                    out = op_f(input_tensor)
                    sms.append(mem_stats(False)-sm)
                    out1 = out + out
                    sms.append(mem_stats(False)-sm)
                    loss = criterion(out.mean().reshape(1,1), comparison)
                    sms.append(mem_stats(False)-sm)
                    loss.backward()
                    sms.append(mem_stats(False)-sm)
                sizes.append(max(sms))
            clean(verbose=False)
            op_mems[op] = np.mean(sizes) / 1024 / 1024
        pp = pprint.PrettyPrinter(indent=0)
        pp.pprint(op_mems)
    else:
        # get size of entire model
        with open("pickles/size_test_in.pkl", "rb") as f:
            [n, e_c, add_pattern, prune, kwargs] = pkl.load(f)
        data, dim = load_data(kwargs['batch_size'], kwargs['dataset']['name'])
        print(kwargs)
        metric = 1
        model = Net(dim=dim,
                    classes=kwargs['dataset']['classes'],
                    dataset_name=kwargs['dataset']['name'],
                    scale=kwargs['scale'],
                    patterns=kwargs['patterns'],
                    num_patterns=n,
                    metric=metric,
                    total_patterns=kwargs['total_patterns'],
                    random_ops={'e_c': e_c, 'i_c': 1.},
                    nodes=kwargs['nodes'],
                    depth=kwargs['depth'],
                    drop_prob=.3,
                    lr_schedule=kwargs['lr_schedule'],
                    prune=True)
        
        model.data = data

        if kwargs.get('remove_prune') is True:
            model.remove_pruners(remove_input=True, remove_edge=True)
            model.add_pattern(full_ops=True)
            print(model)
        elif add_pattern:
            model.add_pattern(prune=prune)
            print(model)
        if kwargs.get('detail', False):
            model.detail_print()
        if 1: #kwargs.get('print_model', False):
            print(model)
        out = list(size_test(model, verbose=kwargs.get('verbose', False)))
        print(out)
        out.append(model.genotype_compression(used_ratio=True)[0])
        edge_counts = collections.Counter()
        
        for cell in model.cells:
            for key, edge in cell.edges.items():
                new_dim = tuple(list(edge.dim) + [edge.stride])
                for k, op in commons.items():
                    edge_counts[(new_dim, k)] = 0 
        
        for cell in model.cells:
            for key, edge in cell.edges.items():
                new_dim = tuple(list(edge.dim) + [edge.stride])
                for op in edge.ops:
                    edge_counts[(new_dim, op.name)] += 1
        out.append(edge_counts)
        out.append(str(model))
        with open("pickles/size_test_out.pkl", "wb") as f:
            pkl.dump(out, f)
