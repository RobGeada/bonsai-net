from bonsai.data_loaders import load_data
from bonsai.net import Net
from bonsai.trainers import *
from bonsai.helpers import *


mem_stats()

if __name__ == '__main__':
    nas_schedule = {'learn_phase': 16,
                    'prune_phase': 16,
                    'prune_interval': 4}
    hypers = {
        'gpu_space': 8.25,
        'dataset': 'CIFAR10',
        'classes': 10,
        'batch_size': 64,
        'scale': 5,
        'nodes': 4,
        'patterns': [['n', 'n', 'na'], ['r', 'n', 'n', 'na'], ['r', 'n', 'na'], ['r', 'n', 'na'], ['r', 'n', 'na'],
                     ['r', 'na'], ['n', 'na']],
        'half': False,
        'multiplier': 1,
        'lr_schedule':
            {'lr_max': .01,
             'T': 600},
        'drop_prob': .25,
        'prune_rate': {'edge': .5, 'input': .5}
    }
    data, dim = load_data(hypers['batch_size'], hypers['dataset'])
    hypers['num_patterns'] = 7

    model = Net(dim=dim,
                classes=hypers['classes'],
                scale=hypers['scale'],
                num_patterns=hypers['num_patterns'],
                patterns=hypers['patterns'],
                nodes=hypers['nodes'],
                random_ops={'e_c': .5, 'i_c': 1},
                drop_prob=hypers['drop_prob'],
                lr_schedule=hypers['lr_schedule'],
                prune=False)
    model.data = data
    print(model)
    full_train(model, hypers['lr_schedule']['T'])