import argparse
import random
from itertools import combinations
from random import choice

import numpy as np
import pandas as pd
import torch

from trainer import GNNTrainer


def run_gnn_search(hyperparams):

    FIRST = True

    layers = ['gcnconv', 'gatconv', 'graphconv', 'tagconv', 'sgconv', 'armaconv']
    layer1combinations = list(combinations(layers,1)) + list(combinations(layers,2)) + list(combinations(layers,3))

    for layer1 in layer1combinations:
        for layer2 in layers:
            for act in ['relu', 'leakyrelu', 'tanh', 'sigmoid', 'elu', 'softplus']:
                for skips in [['no','no'], ['no','yes'], ['yes','no'], ['yes','yes']]:
                    for merge in ['add', 'concat']:
                        if len(layer1) == 1 and merge == 'concat':
                            continue # no need to repeat single first layer architectures with different merge type since there is no merging here
                        for size in [8, 16, 32, 64, 128, 256]:
                            for seed in [81, 1458, 1729]:

                                encoding = {'layertype1':layer1, 'layertype2':layer2, 'act1':act, 'width1':size, 'skips':skips, 'merge':merge}

                                np.random.seed(seed)
                                torch.manual_seed(seed)
                                torch.cuda.manual_seed(seed)

                                t = GNNTrainer(encoding, hyperparams, dataset_name=args.dataset, gpuid=args.gpuid)
                                results = t.train()

                                results['encoding'] = str(encoding)
                                results['seed'] = seed
                                
                                df = pd.DataFrame.from_dict({key:[value] for key, value in results.items()})
                                print(encoding)
                                print(df)
                                
                                if FIRST: # first run, write column headings
                                    df.to_csv(f'gnn_search_{args.dataset}.csv', index=False, header=True, mode='w')
                                    FIRST = False
                                else:
                                    df.to_csv(f'gnn_search_{args.dataset}.csv', index=False, header=False, mode='a')

    return results


def run_hp_search():

    FIRST = True
    encoding_list = []
    for i in range(50):
        # make random encoding:
        def randlayer1():
            layers = ['gcnconv', 'gatconv', 'graphconv', 'tagconv', 'sgconv', 'armaconv']
            layer1combinations = list(combinations(layers,1)) + list(combinations(layers,2)) + list(combinations(layers,3))
            return choice(layer1combinations)
        def randlayer2():
            return choice(['gcnconv', 'gatconv', 'graphconv', 'tagconv', 'sgconv'])
        def randact():
            return choice(['relu', 'leakyrelu', 'tanh', 'sigmoid', 'elu', 'softplus'])
        def randwidth():
            return choice([8, 16, 32, 64, 128, 256])
        def randskips():
            return [choice(['yes','no']), choice(['yes','no'])]
        def randmerge():
            return choice(['add', 'concat'])

        def randencoding():
            return {'layertype1':randlayer1(), 'layertype2':randlayer2(), 'act1':randact(), 'width1':randwidth(), 'skips':randskips(), 'merge':randmerge()}
        
        encoding = randencoding()

        while encoding in encoding_list:
            print('Model already used - generating a new one!')
            encoding = randencoding()
        encoding_list.append(encoding)

        print(encoding)

        for lr in [0.1 , 0.01, 0.001, 0.0005]:
            for wd in [1e-3, 5e-4, 1e-5, 5e-6]:
                for opt in ['adam', 'sgd', 'rmsprop']:
                    for dropout in [0.0, 0.25, 0.5, 0.75]:
                        for seed in [81, 1458, 1729]:
                            
                            hyperparams = {'lr':lr, 'wd':wd, 'opt':opt, 'dropout':dropout}

                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed(seed)

                            t = GNNTrainer(encoding, hyperparams, dataset_name=args.dataset, gpuid=args.gpuid)
                            results = t.train()

                            results['encoding'] = str(encoding)
                            results['seed'] = seed
                            results.update(hyperparams)
                            
                            df = pd.DataFrame.from_dict({key:[value] for key, value in results.items()})
                            print(encoding)
                            print(hyperparams)
                            print(df)
                            
                            if FIRST: # first run, write column headings
                                df.to_csv(f'hyperparam_gridsearch_{args.dataset}.csv', index=False, header=True, mode='w')
                                FIRST = False
                            else:
                                df.to_csv(f'hyperparam_gridsearch_{args.dataset}.csv', index=False, header=False, mode='a')

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'], help='Dataset to use')
    parser.add_argument('--gpuid', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--mode', type=str, choices=['hp', 'ss'])
    args = parser.parse_args()

    if args.mode == 'hp':
        results = run_hp_search()
    elif args.mode == 'ss':
        hyperparams = {'lr':args.lr, 'wd':args.wd, 'opt':args.opt, 'dropout':args.dropout}
        results = run_gnn_search(hyperparams)
