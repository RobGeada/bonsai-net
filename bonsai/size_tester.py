import os, sys

os.chdir(os.getcwd()+"/../")
sys.path.append(os.getcwd())
print(os.getcwd())

import pickle as pkl
from bonsai.data_loaders import load_data
from bonsai.net import Net
from bonsai.trainers import size_test



if __name__=='__main__':
        with open("pickles/size_test_in.pkl","rb") as f:
            [n,e_c,prune,kwargs]=pkl.load(f)

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
            print(size_test(model, verbose=kwargs.get('verbose',False)))
            model.remove_pruners(remove_edge=True)
            print(size_test(model, verbose=kwargs.get('verbose', False)))
            model.add_pattern(full_ops=True)
        elif kwargs.get('add_pattern', False):
            model.add_pattern(prune=prune, full_ops=True)
        if kwargs.get('detail', False):
            model.detail_print()
        if kwargs.get('print_model', False):
            print(model)
        out = size_test(model, verbose=kwargs.get('verbose',False))
        with open("pickles/size_test_out.pkl","wb") as f:
            pkl.dump(out,f)