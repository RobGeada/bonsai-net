import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ELU, LeakyReLU, ReLU, Sigmoid, Tanh, Softplus

from torch_geometric.nn import GATConv, GCNConv, GraphConv, TAGConv, SGConv, ARMAConv
from bonsai_gcn.pruner import *
from bonsai_gcn.helpers import *

class PrunerEdge(torch.nn.Module):
    def __init__(self, ops, name, init, prune=True):
        super().__init__()
        self.name = name
        self.prune = prune
        self.edge = nn.ModuleDict({'pruner':Pruner(init=init), 'op': nn.ModuleList(ops)})
        
    def forward(self, x):
        for i,op in enumerate(self.edge['op']):
            x = op(*x) if i==0 else op(x)
        if self.prune:
            return self.edge['pruner'](x)
        else:
            return x

class SpikySig(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = 1e5
        self.saw = lambda w: (self.m * w - torch.floor(self.m * w)) / self.m
        self.act = Sigmoid()
        
    def forward(self,x):
        return self.act(x) + self.saw(x)
    
class SpikyRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = 1e5
        self.saw = lambda w: (self.m * w - torch.floor(self.m * w)) / self.m
        self.act = ReLU()
        
    def forward(self,x):
        return self.act(x) + self.saw(x)
    
class MasterModel(torch.nn.Module):
    def __init__(self, dataset, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        layer_map = {'gcnconv':GCNConv, 
                     'gatconv':MyGAT, 
                     'graphconv':GraphConv, 
                     'tagconv': TAGConv, 
                     'sgconv':SGConv, 
                     'armaconv':ARMAConv}

        act_map = {'relu':ReLU, 
                   'leakyrelu':LeakyReLU, 
                   'tanh':Tanh, 
                   'sigmoid':Sigmoid, 
                   'elu':ELU, 
                   'softplus':Softplus}
        
        self.num_layers = len(layer_map)
        self.num_acts = len(act_map)
        
        if kwargs.get("config",None) is not None:
            config = kwargs['config']
            if type(config['layer1'][0]) is list:
                layer_perm = [((l_name, a_name),(layer_map[l_name],act_map[a_name]))
                              for l_name, a_name in config['layer1']]
            else:
                act_map = {k:v for k,v in act_map.items() if k in config['act']}
            layer1_map = {k:v for k,v in layer_map.items() if k in config['layer1']}
            layer2_map = {k:v for k,v in layer_map.items() if k in config['layer2']}
            add_skip1 = config['skip1']
            add_skip2 = config['skip2']
            
        else:
            layer1_map = layer_map
            layer2_map = layer_map
            layer_perm = [((l_name,a_name), (layer,act)) 
                          for l_name, layer in layer_map.items() for a_name, act in act_map.items()]
            add_skip1 = True
            add_skip2 = True
        
            
        prune = kwargs.get('prune', True)
        self.prune = prune
        p_weight = kwargs['p_weight']
        
        if kwargs['merge_acts'] is False:
            layer1 = [PrunerEdge([layer(dataset.num_node_features, kwargs['width'])], 
                                 name, 
                                 init=p_weight,
                                 prune=(prune and len(layer1_map)>1)) 
                      for name, layer in layer1_map.items()]
            act = [PrunerEdge([act()],
                              name, 
                              init=p_weight,
                              prune=(prune and len(act_map)>1))
                   for name, act in act_map.items()]
            self.act = nn.ModuleList(act)
        else:
            layer1 = []
            for (l_name,a_name), (layer, act) in layer_perm:
                 layer1.append(
                     PrunerEdge(
                         [layer(dataset.num_node_features, kwargs['width']), act()],
                         name=[l_name, a_name],
                         init=p_weight,
                         prune=(prune and len(layer_perm)>1)))
            self.act = None
        self.layer1 = nn.ModuleList(layer1)
        
  
        layer2 = [PrunerEdge([layer(kwargs['width'], dataset.num_classes)], 
                             name, 
                             init=p_weight,
                             prune=(prune and len(layer2_map)>1)) 
                  for name, layer in layer2_map.items()]
        self.layer2 = nn.ModuleList(layer2)
        
        if add_skip1:
            skip1 = nn.Linear(dataset.num_node_features, kwargs['width'])
            self.skip1 = PrunerEdge([skip1], 'Skip1', init=p_weight, prune=prune)
        else:
            self.skip1 = False
        
        if add_skip2:
            skip2 = nn.Linear(kwargs['width'], dataset.num_classes)
            self.skip2 = PrunerEdge([skip2], 'Skip2', init=p_weight, prune=prune)
        else:
            self.skip2 = False
    
    def get_pruners(self, join=True):
        l1 = [edge.edge['pruner'] for edge in self.layer1 if edge.prune]
        if self.act is not None:
            a1 = [edge.edge['pruner'] for edge in self.act if edge.prune]
        else:
            a1 = []
        l2 = [edge.edge['pruner'] for edge in self.layer2 if edge.prune]
        s1 = [self.skip1.edge['pruner']] if self.skip1 else []
        s2 = [self.skip2.edge['pruner']] if self.skip2 else []
        
        if join:
            return l1+a1+l2+s1+s2
        else:
            return ('l1',l1),('a1',a1),('l2',l2),('s1',s1),('s2',s2)
    
    def track_gates(self, deadhead, epoch, verbose=True, print_weights=False):
        dhs = [pruner.track_gates(deadhead, print_weights=print_weights) for pruner in self.get_pruners()]
        remaining = sum([1 for pruner in self.get_pruners() if not pruner.deadheaded])
            
        if verbose and sum(dhs):
            print("E{}: Deadheaded {} edges. {} remaining.".format(epoch, sum(dhs), remaining))
        
        layer1 = [edge for edge in self.layer1 if not edge.edge['pruner'].deadheaded and edge.prune]
        if len(layer1)==1:
            print("Switching off layer1 pruner")
            layer1[0].prune = False
        
        if self.act is not None:
            act1 = [edge for edge in self.act if not edge.edge['pruner'].deadheaded and edge.prune]
            if len(act1)==1:
                print("Switching off act pruner")
                act1[0].prune = False
            
        layer2 = [edge for edge in self.layer2 if not edge.edge['pruner'].deadheaded and edge.prune]
        if len(layer2)==1:
            print("Switching off layer2 pruner")
            layer2[0].prune = False
    
    def get_config(self):           
        layer1 = [edge.name for edge in self.layer1 if not edge.edge['pruner'].deadheaded]
        if self.act is not None:
            act = [edge.name for edge in self.act if not edge.edge['pruner'].deadheaded]
        else:
            act = None
        layer2 = [edge.name for edge in self.layer2 if not edge.edge['pruner'].deadheaded]
        skip1 = self.skip1.edge['pruner'].deadheaded if self.skip1 else False
        skip2 = self.skip2.edge['pruner'].deadheaded if self.skip1 else False
        return {'layer1': layer1, 'act':act, 'layer2':layer2,'skip1':skip1,'skip2':skip2}
        
    def print(self):
        print("Layer 1:",[(edge.name,edge.edge['pruner'].sg().item()) for edge in self.layer1 if not edge.edge['pruner'].deadheaded])
        if self.act is not None:
            print("Act 1  :",[(edge.name,edge.edge['pruner'].sg().item()) for edge in self.act if not edge.edge['pruner'].deadheaded])
        print("Skip 1 :",'Yes' if self.skip1 and not self.skip1.edge['pruner'].deadheaded else 'No')
        print("Layer 2:",[(edge.name,edge.edge['pruner'].sg().item()) for edge in self.layer2 if not edge.edge['pruner'].deadheaded])
        print("Skip 2 :",'Yes' if self.skip2 and not self.skip2.edge['pruner'].deadheaded else 'No')
        
    def forward(self, data, verbose=False):
        x, edge_index = data.x, data.edge_index
        # x: V x H matrix of node features
        # edge_index: V x V adjacency matrix (sparse)

        # first layer:
        layer1_outs = sum([layer([x, edge_index]) for layer in self.layer1])
        if self.act is not None:
            x = sum([act([layer1_outs]) for act in self.act])
        else:
            x = layer1_outs

        # first skip connection if present:
        if self.skip1:
            x += self.skip1([data.x])
        
        # cache layer 1 activations for second skip connection if used:
        if self.skip2:
            identity = x
        
        # dropout:
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # second layer:
        x = sum([layer([x, edge_index]) for layer in self.layer2])

        # second skip connection, if present:
        if self.skip2:
            x += self.skip2([identity])

        if verbose:
            print(cache_stats())
        return x

class MyGAT(GATConv):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, heads=8, concat=False)