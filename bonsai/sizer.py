import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../")))

import pickle as pkl
import pprint
import torch

from bonsai.data_loaders import load_data
from bonsai.helpers import mem_stats, clean
from bonsai.net import Net
from bonsai.ops import commons
from bonsai.trainers import size_test


if __name__ == '__main__':
    if len(sys.argv)> 1 and sys.argv[1]=='o':
        # get sizes of individual operations in network
        # read args
        C = int(sys.argv[3])
        dim = [int(x) for x in sys.argv[2:]]

        # build op
        op_mems = {}
        input = torch.zeros(dim).cuda()

        for op, f in commons.items():
            start_mem = mem_stats(False)
            op_f = f(C, 1).cuda()
            out = op_f(input)
            end_mem = mem_stats(False) - start_mem
            del out, op_f
            clean(verbose=False)
            op_mems[op] = end_mem / 1024 / 1024
        pp = pprint.PrettyPrinter(indent=0)
        pp.pprint(op_mems)

    else:
        # get size of entire model
        with open("pickles/size_test_in.pkl", "rb") as f:
            [n, e_c, prune, kwargs] = pkl.load(f)

        data, dim = load_data(kwargs['batch_size'], kwargs['dataset'])
        model = Net(dim=dim,
                    classes=kwargs['classes'],
                    scale=kwargs['scale'],
                    patterns=kwargs['patterns'],
                    num_patterns=n,
                    random_ops={'e_c': e_c, 'i_c': 1.},
                    nodes=kwargs['nodes'],
                    drop_prob=.3,
                    lr_schedule=kwargs['lr_schedule'],
                    prune=True)
        model.data = data

        if kwargs.get('remove_prune') is True:
            print(size_test(model, verbose=kwargs.get('verbose', False)))
            model.remove_pruners(remove_edge=True)
            print(size_test(model, verbose=kwargs.get('verbose', False)))
            model.add_pattern(full_ops=True)
        elif kwargs.get('add_pattern', False):
            model.add_pattern(prune=prune, full_ops=True)
        if kwargs.get('detail', False):
            model.detail_print()
        if kwargs.get('print_model', False):
            print(model)
        out = size_test(model, verbose=kwargs.get('verbose', False))
        with open("pickles/size_test_out.pkl", "wb") as f:
            pkl.dump(out, f)
