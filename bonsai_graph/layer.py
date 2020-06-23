from bonsai_graph.ops import *
from bonsai_graph.pruner import *
  
import torch
import torch.nn as nn
import torch.nn.functional as F


def gat_message(edges):
    if 'norm' in edges.src:
        msg = edges.src['ft'] * edges.src['norm']
        return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1'], 'norm': msg}
    return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1']}


class NASLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # operations within layer
        self.attention_type = kwargs['att']
        self.aggregator_type = kwargs['agg']
        self.act = kwargs['act']
        self.name = '{}_{}_{}'.format(self.aggregator_type, 
                                      self.attention_type, 
                                      self.act)
        
        # layer size
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.num_heads = int(kwargs['num_heads'])
        
        # layer characteristics
        self.concat = kwargs['concat']
        self.dropout = kwargs.get('dropout', .6)
        self.residual = kwargs['residual']
        self.attention_dim = 64 if self.attention_type in ['cos', 'generalized_linear'] else 1
        self.pooling_dim = kwargs.get('pooling_dim', 128)
        self.batch_normal = kwargs.get('batch_normal',True)

        # layer modules
        self.bn = nn.BatchNorm1d(self.in_channels, momentum=0.5)
        self.prp = nn.ModuleList()
        self.red = nn.ModuleList()
        self.fnl = nn.ModuleList()
        self.agg = nn.ModuleList()
        for hid in range(self.num_heads):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.prp.append(AttentionPrepare(self.in_channels, self.out_channels, self.attention_dim, self.dropout))
            agg = aggs[self.aggregator_type](self.out_channels, self.pooling_dim)
            self.agg.append(agg)
            self.red.append(attents[self.attention_type](self.dropout, self.attention_dim, agg))
            self.fnl.append(GATFinalize(hid, self.in_channels, self.out_channels, acts[self.act], self.residual))


    def forward(self, features, g):
        last = self.bn(features) if self.batch_normal else features

        for hid in range(self.num_heads):
            # prepare
            g.ndata.update(self.prp[hid](last))
            # message passing
            g.update_all(gat_message, self.red[hid], self.fnl[hid])
            
        # merge all the heads
        if not self.concat:
            output = g.pop_n_repr('head0')
            for hid in range(1, self.num_heads):
                output = torch.add(output, g.pop_n_repr('head%d' % hid))
            output = output / self.num_heads
        else:
            output = torch.cat([g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads)], dim=1)
        del last
        return output
    
    
class MultiLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.prune = kwargs['prune']
        self.out_size = kwargs['out_channels']*kwargs['num_heads'] if kwargs['concat'] else kwargs['out_channels']
        
        # build all permutations of all layers if configuration not specified
        layers = []
        if kwargs.get('ops_by_layer') is None:
            for act in acts.keys():
                for att in attents.keys():
                    for agg in aggs.keys():
                        op = NASLayer(att=att,
                                      agg=agg, 
                                      act=act,
                                      num_heads=kwargs['num_heads'],
                                      in_channels=kwargs['in_channels'],
                                      out_channels=kwargs['out_channels'],
                                      concat=kwargs['concat'],
                                      residual=kwargs['residual'])
                        p = Pruner(init=kwargs['init']/2) if self.prune else None
                        layers.append(nn.ModuleDict({'op':op,'pruner':p}))
        # otherwise, build only specific permutations
        else:
            for op in kwargs['ops_by_layer']:
                op = NASLayer(att=op['att'], 
                              agg=op['agg'], 
                              act=op['act'], 
                              num_heads=kwargs['num_heads'],
                              in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              concat=kwargs['concat'],
                              residual=kwargs['residual'])
                p = Pruner(init=kwargs['init']/2) if self.prune and len(kwargs['ops_by_layer'])>1 else None
                layers.append(nn.ModuleDict({'op':op,'pruner':p}))
        self.layers = nn.ModuleList(layers)
            
        
    def forward(self, features, g):
        out = torch.zeros((features.shape[0], self.out_size), 
                          dtype=features.dtype, 
                          device=features.device)
        for i, layer in enumerate(self.layers):
            if self.prune and layer['pruner'] is not None:
                if not layer['pruner'].deadheaded:
                    out += layer['pruner'](layer['op'](features, g))
            else:
                out += layer['op'](features, g)
        return out

