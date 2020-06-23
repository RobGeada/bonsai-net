import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from bonsai_gcn.mastermodel import MasterModel

# manager the train process of GNN on citation dataset
class GNNTrainer(object):
    def __init__(self,  kwargs, gpuid=0):

        # encoding = {'layer_types':['gcnconv','gcnconv'], 'acts':['sigmoid'], 'widths':[32]}
        dataset_name = kwargs['dataset_name']
        assert dataset_name in ['Cora', 'CiteSeer', 'PubMed']
        self.num_epochs = kwargs['epochs']
        self.comp_lambda = kwargs['comp_lambda']
        self.device = f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu'
        if 'cuda' in self.device:
            torch.cuda.set_device(f'cuda:{gpuid}')

        
        self.dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
        self.model = MasterModel(self.dataset, **kwargs).to(self.device)
        self.data = self.dataset[0].to(self.device)

        opt_class = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD, 'rmsprop':torch.optim.RMSprop}
        if kwargs['opt']=='sgd':
            self.optimizer = opt_class[kwargs['opt']](self.model.parameters(), 
                                                      lr=kwargs['lr'], 
                                                      momentum=.9,
                                                      weight_decay=kwargs['wd'])
        else:
            self.optimizer = opt_class[kwargs['opt']](self.model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['wd'])
        self.xentropy = torch.nn.CrossEntropyLoss()
    
    def comp_loss(self):
        ratios = []
        targets = []
        layer_perm = self.model.act is None
        for name, p_set in self.model.get_pruners(join=False):
            if len(p_set)>1:
                ops = [pruner.sg() for pruner in p_set if not pruner.deadheaded]
                if len(ops):
                    if layer_perm and name=='l1':
                        targets.append(1/(self.model.num_layers*self.model.num_acts))
                    elif not layer_perm or 'l' in name:
                        targets.append(1/(self.model.num_layers))
                    elif 'a' in name:
                        targets.append(1/self.model.num_acts)
                    ratios.append(sum(ops)/len(ops))
        if len(ratios)==0:
            return torch.tensor(0), False
        ratios = torch.cat(ratios,0)
        targets = torch.tensor(targets, device=ratios.device)
        return self.comp_lambda * torch.dist(ratios,targets), True
    
    def train(self, verbose=True, print_weights=False):
        num_params = sum([x.numel() for x in self.model.parameters()])
        self.model.train()
        results = {'params':num_params}
        results['train_acc'] = []
        results['train_loss']= []
        results['eval_acc'] = []
        results['test_acc'] = []
        print('Model has {:,} parameters'.format(num_params))
        start_time = time.time()
        for epoch in range(self.num_epochs+1):
            
            self.model.train()
            self.optimizer.zero_grad()

            out = self.model(self.data, epoch==0)
            self.model.track_gates(epoch=epoch, deadhead=16, print_weights=print_weights)
            
            
            class_loss = self.xentropy(out[self.data.train_mask], self.data.y[self.data.train_mask])        
            if self.model.prune:
                comp_loss = self.comp_loss()
                if comp_loss[-1]:
                    loss = class_loss+comp_loss[0]
                else:
                    loss = class_loss
            else:
                loss = class_loss
                comp_loss = torch.tensor(0.), False
            
            # Call the eval function at some interval
            # calculate current train accuracy:
            pred = out.argmax(dim=1)
            correct_train = float(pred[self.data.train_mask].eq(self.data.y[self.data.train_mask]).sum().item())
            train_acc = correct_train / self.data.train_mask.sum().item()
            results['train_acc'].append(round(train_acc, 5))
            results['train_loss'].append(round(loss.item(), 5))

            if epoch%10==0 and verbose:
                print("==EPOCH {}==".format(epoch))
                print("Class loss: {:.2f}, Comp loss: {:.2f}".format(class_loss.item(), comp_loss[0].item()))
            eval_acc, test_acc = self.evaluate(verbose=verbose and (epoch%50==0))
            results['eval_acc'].append(eval_acc)
            results['test_acc'].append(test_acc)

            loss.backward()
            self.optimizer.step()
        
        total_time = time.time() - start_time
        print(f"Training complete in {total_time} seconds")
        results['run_time'] = total_time

        self.results = results # just in case
        
        return results

    def evaluate(self, verbose):
        
        self.model.eval()
        out = self.model(self.data)
        pred = out.argmax(dim=1)

        correct_eval = float (pred[self.data.val_mask].eq(self.data.y[self.data.val_mask]).sum().item())
        eval_loss = self.xentropy(out[self.data.val_mask], self.data.y[self.data.val_mask]).item()
        eval_acc = correct_eval / self.data.val_mask.sum().item()

        correct_test = float (pred[self.data.test_mask].eq(self.data.y[self.data.test_mask]).sum().item())
        test_loss = self.xentropy(out[self.data.test_mask], self.data.y[self.data.test_mask]).item()
        test_acc = correct_test / self.data.test_mask.sum().item()

        if verbose:
            print('  Eval Accuracy: {:.4f}'.format(eval_acc))
            print('  Test Accuracy: {:.4f}'.format(test_acc))

        return round(eval_acc, 5), round(test_acc, 5)

    def reset(self, encoding):
        """Create a new model with a given encoding and reset the optimiser"""

        self.encoding = encoding

        self.model = MasterModel(self.encoding, self.dataset, self.hyperparams['dropout']).to(self.device)
        opt_class = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD, 'rmsprop':torch.optim.RMSprop}
        self.optimizer = opt_class[self.hyperparams['opt']](self.model.parameters(), lr=self.hyperparams['lr'], weight_decay=self.hyperparams['wd'])

