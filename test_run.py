from bonsai.nas import Bonsai
from bonsai.helpers import mem_stats

mem_stats()

hypers = {
    'gpu_space': 10.25,
    'dataset':{'name':'CIFAR10', 'classes':10},
    'batch_size':64,
    'scale':32,
    'nodes':7,
    'patterns': [['na'], ['r']],
    'reduction_target':2,
    'lr_schedule': {'lr_max': .01, 'T': 600},
    'drop_prob': .3,
    'nas_schedule': {'prune_interval':4, 'cycle_len':8},
    'prune_rate':{'edge':.01, 'input':.01}
}

# sizes={
#     1:0.8928571428571428,
#     2:0.46428571428571425,
#     3:0.5446428571428571,
#     4:0.4107142857142857,
# }
# start_size=1
# bonsai = Bonsai(hypers, sizes=sizes, start_size=start_size)

bonsai = Bonsai(hypers)
bonsai.train()
bonsai.random_search(1)
bonsai.random_search(3)
