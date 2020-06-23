import os
import time
import collections
from collections import namedtuple

import numpy as np
import pickle as pkl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data

from bonsai_graph.graphnet import GraphNet
from bonsai_graph.ops import *
from bonsai_graph.helpers import *

def load(kwargs, save_file=".pkl"):
    kwarg_nt = namedtuple('kwarg', kwargs.keys())(*kwargs.values())

    save_file = 'data/' + kwarg_nt.dataset + save_file 
    if os.path.exists(save_file):
        with open(save_file, "rb") as f:
            return pkl.load(f)
    else:
        datas = load_data(kwarg_nt)
        with open(save_file, "wb") as f:
            pkl.dump(datas, f)
        return datas


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


def compression_loss(model, comp_lambda):
    nums, dens = [], []
    w = model.layers[0].layers[0]['pruner'].weight
    zero = torch.tensor([0.], device=w.device)
    for multilayer in model.layers:
        pruners = [layer['pruner'].sg() for layer in multilayer.layers]
        dens += [len(pruners)]
        nums += [sum(pruners)]
    nums = torch.cat(nums)
    dens = torch.tensor(dens, device=w.device, dtype=w.dtype)
    comp_ratio = torch.div(nums, dens)
    print(1/dens, comp_ratio)
    comp = torch.norm(1/dens - comp_ratio)
    loss = comp_lambda*comp
    return loss


# manager the train process of GNN on citation dataset
class CitationGNNManager(object):
    def __init__(self, kwargs):

        self.kwargs = kwargs

        if  kwargs['dataset'] in ["cora", "citeseer", "pubmed"]:
            self.data = load(kwargs)
            self.kwargs['in_feats'] = self.in_feats = self.data.features.shape[1]
            self.kwargs['num_class'] = self.n_classes = self.data.num_labels

        self.kwargs = kwargs
        
        # training dynamics
        self.drop_out = kwargs.get('in_drop',.6)
        self.multi_label = kwargs.get('multi_label', False)
        self.lr = kwargs.get('lr', .005)
        self.comp_lambda = kwargs.get('comp_lambda',.01)
        self.weight_decay = kwargs.get('weight_decay', 5e-4)
        self.loss_fn = torch.nn.functional.nll_loss
        self.epochs = kwargs.get('epochs', 300)
        
        # model size and shape
        self.num_layers  = self.kwargs['num_layers']
        self.num_heads   = self.kwargs['num_heads']
        self.num_hiddens = self.kwargs['num_hiddens']
        self.prune = self.kwargs['prune']
        self.ops_by_layer = self.kwargs.get('ops_by_layer',[None for _ in range(self.num_layers)])
        self.cuda = True
        
        self.build_model()

    def build_model(self):
        self.model = GraphNet(num_feats=self.in_feats, 
                              num_labels=self.n_classes, 
                              num_layers=self.num_layers, 
                              num_heads=self.num_heads, 
                              num_hiddens=self.num_hiddens,
                              prune=self.prune, 
                              lr=self.lr,
                              ops_by_layer=self.ops_by_layer, 
                              drop_out=self.drop_out, 
                              multi_label=False, 
                              batch_normal=False)
        if self.cuda:
            self.model.cuda()

    def run_model(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        dur = []
        begin_time = time.time()
        best_performance = 0
        max_val_acc = 0
        min_train_loss = float("inf")
        model_val_acc = 0
        features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNNManager.prepare_data(self.data, self.cuda)
        mask = mask.to(torch.bool)
        val_mask = val_mask.to(torch.bool)
        test_mask = test_mask.to(torch.bool)
        c = collections.Counter([x.item() for x in labels[test_mask]])
        print({k:v/sum(c.values()) for k,v in c.items()})

        for epoch in range(0, self.epochs):            
            self.model.train()
            
            t0 = time.time()
            # forward
            logits = self.model(features, g, verbose=epoch==0)
            size = mem_stats(False)/1024/1024/1024
            logits = F.log_softmax(logits, 1)
            loss = self.loss_fn(logits[mask], labels[mask])
            if self.prune:
                self.model.track_pruners()
                comp_loss = compression_loss(self.model, self.comp_lambda)
                loss += comp_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #annealer.step()
            train_loss = loss.item()
            del logits

            # evaluate
            self.model.eval()
            logits = self.model(features, g)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, labels, mask)
            dur.append(time.time() - t0)

            val_loss = float(self.loss_fn(logits[val_mask], labels[val_mask]))
            val_acc = evaluate(logits, labels, val_mask)
            test_acc = evaluate(logits, labels, test_mask)
            del logits
            
            if epoch%8==0 and self.prune:
                self.model.deadhead()
                self.model.print_ops(lens=True)
            

            if val_acc > max_val_acc:  # and train_loss < min_train_loss
                max_val_acc = val_acc
                min_train_loss = train_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc
            if True:
                if self.prune:
                    info_str = "Epoch {:05d} | Loss (L: {:.4f}, C: {:.4f}) | Time(s) {:.4f} |"
                    info_str +=" comp {:.2f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}|"
                    print(info_str.format(epoch, 
                                          loss.item(),
                                          comp_loss.item(), 
                                          np.mean(dur), 
                                          self.model.compression(), 
                                          train_acc, 
                                          val_acc, 
                                          test_acc))
                else:
                    info_str = "Epoch {:05d} | Loss (L: {:.4f}) | Time(s) {:.4f} | comp n/a |"
                    info_str += " acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}"
                    print(info_str.format(epoch, 
                                          loss.item(), 
                                          np.mean(dur), 
                                          train_acc, 
                                          val_acc, 
                                          test_acc))
        print("val_score:{},test_score:{}".format(model_val_acc, best_performance))
        if self.prune:
            self.model.deadhead()
        return best_performance, size

    def update_ops(self):
        layer_preserve = [[layer['op'].name for layer in multilayer.layers if not layer['pruner'].deadheaded] for multilayer in self.model.layers]
        ops_by_layer = []
        for layer in layer_preserve:
            layer_ops = []
            new_aggs, new_atts, new_acts = [],[],[]
            for op in layer:
                agg, att, act = op.split("_")
                layer_ops.append({'att':att,'agg':agg,'act':act})
            ops_by_layer.append(layer_ops)
        self.ops_by_layer = ops_by_layer
        
        
    @staticmethod
    def prepare_data(data, cuda=True):
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.ByteTensor(data.train_mask)
        test_mask = torch.ByteTensor(data.test_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        n_edges = data.graph.number_of_edges()
        # create DGL graph
        g = DGLGraph(data.graph)
        # add self loop
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        if cuda:
            features = features.cuda()
            labels = labels.cuda()
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        return features, g, labels, mask, val_mask, test_mask, n_edges