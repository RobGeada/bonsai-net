import torch
import torch.nn as nn
import torch.nn.functional as F

from bonsai_graph.ops import *
from bonsai_graph.layer import *
from bonsai_graph.utils.tensor_utils import cache_stats

class GraphNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # optional args
        self.residual = kwargs.get('residual', True)
        
        # network shape parameters
        self.num_feats   = kwargs['num_feats']
        self.num_labels  = kwargs['num_labels']
        self.num_layers  = kwargs['num_layers']
        self.num_heads   = kwargs['num_heads']
        self.num_hiddens = kwargs['num_hiddens']
        
        # operation dynamics
        self.prune = kwargs['prune']
        self.lr = kwargs['lr']
        self.ops_by_layer = kwargs['ops_by_layer']

        # build layers
        self.layers = nn.ModuleList()
        self.build_hidden_layers()
    
    def print_ops(self, lens):
        for i, multilayer in enumerate(self.layers):
            if lens:
                print("{}: {}".format(i, len([layer for layer in multilayer.layers if not layer['pruner'].deadheaded])))
            else:
                print("{}: {}".format(i, [layer['op'].name for layer in multilayer.layers if not layer['pruner'].deadheaded]))
        
    def build_hidden_layers(self):
        for i in range(self.num_layers):
            in_channels = self.num_feats if i == 0 else out_channels * self.num_heads 
            concat = i != self.num_layers - 1
            out_channels = self.num_labels if not concat else self.num_hiddens
            residual = False and self.residual if i == 0 else True and self.residual
            
            # create layer
            self.layers.append(MultiLayer(num_heads=self.num_heads, 
                                          in_channels=in_channels,
                                          out_channels=out_channels, 
                                          ops_by_layer=self.ops_by_layer[i], 
                                          concat=concat, 
                                          init=self.lr,
                                          residual=residual,
                                          prune=self.prune))

    def track_pruners(self):
        [layer['pruner'].track_gates() for multilayer in self.layers for layer in multilayer.layers if layer['pruner'] is not None]

        
    def deadhead(self):
        deadheads = [layer['pruner'].deadhead() for multilayer in self.layers for layer in multilayer.layers if layer['pruner'] is not None]
        remaining = sum([not layer['pruner'].deadheaded for multilayer in self.layers for layer in multilayer.layers if layer['pruner'] is not None])
        print("Deadheaded {} ops. Remaining: {}".format(sum(deadheads),remaining)) 

            
    def compression(self):
        pruners = [layer['pruner'].sg() for multilayer in self.layers for layer in multilayer.layers if layer['pruner'] is not None]
        return sum(pruners).detach().item()/len(pruners)
            
    def forward(self, feat, g, verbose=False):
        output = feat
        for i, layer in enumerate(self.layers):
            output = layer(output, g)
            if verbose:
                print("{}: {}".format(i,cache_stats()))
        return output
