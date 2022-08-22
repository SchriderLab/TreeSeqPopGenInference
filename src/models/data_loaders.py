# -*- coding: utf-8 -*-
import numpy as np
from torch_geometric.data import Data, Batch, DataLoader
import torch
import random

import glob
import os
import copy

class TreeSeqGenerator(object):
    def __init__(self, ifile, models=None, n_samples_per = 16, sequence_length = 32, sequential = True):
        if models is None:
            self.models = list(ifile.keys())
            print(self.models)
        else:
            self.models = models

        # hdf5 file we are reading from
        self.ifile = ifile

        # how many tree sequences from each demographic model are included in a batch?
        self.n_samples_per = n_samples_per
        
        # how many trees in a sequence
        self.s_length = sequence_length

        # if False, trees have random placements on the chromosome        
        self.sequential = sequential

        # shuffle the keys / make sure they are all there (keys are deleted
        # during training or validation so as not to repeat samples)
        self.keys = {model: list(self.ifile[model].keys()) for model in self.models}
        self.on_epoch_end()
        
        return
    
    def __getitem__(self, index):
        # features, edge_indices, and label, what sequence the graphs belong to
        X = []
        X1 = [] # tree-level features (same size as batch_)
        indices = []
        y = []
        batch_ = []

        ij = 0
        for model in self.models:
            # grab n_samples_per for each model
            for ix in range(self.n_samples_per):
                while True:
                    if self.counts[model] == len(self.keys[model]):
                        break
                    
                    model_index = self.models.index(model)
                    key = self.keys[model][self.counts[model]]
    
                    self.counts[model] += 1
                    skeys = sorted(list(self.ifile[model][key].keys()))
                    
                    if not 'x' in skeys:
                        continue
                    
                    X_ = np.array(self.ifile[model][key]['x'])   
                    #print(X_.shape)
                    if len(X_) > self.s_length:
                        ii = np.random.choice(range(len(X_) - self.s_length))
                        ii = range(ii, ii + self.s_length)
                        
                        break
                    
                if self.counts[model] == len(self.keys[model]):
                    return None, None, None
                
                edges = np.array(self.ifile[model][key]['edge_index'], dtype = np.int32)
                
                for ii_ in ii:
                    x = X_[ii_]
                    
                    ik = list(np.where(x[:,0] != 0))
                    x[ik,0] = (np.log(x[ik,0]) - 0.6568706730096306) / 2.5765864628157367
                    
                    X.append(x)
                    indices.append(edges[ii_])
    
                    batch_.append(ij)
                    
                y.append(model_index)
                ij += 1

        if len(y) < 2:
            return None, None, None

        y = torch.LongTensor(np.hstack(y).astype(np.float32))
    
        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k])) for k in range(len(indices))])

        return batch, y, batch_
                
    def on_epoch_end(self):
        counts = dict()
        for model in self.models:
            counts[model] = 0

        for key in self.keys.keys():
            random.shuffle(self.keys[key])
            print(len(self.keys[key]))
            
    def __len__(self):
        return int(np.floor(np.min([len(self.keys[u]) for u in self.keys.keys()]) / self.n_samples_per))

class TreeGenerator(object):
    def __init__(self, ifile, models=None, n_samples_per=5):
        if models is None:
            self.models = list(ifile.keys())
        else:
            self.models = models

        # hdf5 file we are reading from
        self.ifile = ifile

        # how many tree sequences from each demographic model are included in a batch?
        self.n_samples_per = n_samples_per



        # shuffle the keys / make sure they are all there (keys are deleted
        # during training or validation so as not to repeat samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(np.min([len(self.keys[u]) for u in self.keys.keys()]) / self.n_samples_per))

    def __getitem__(self, index):
        # features, edge_indices, and label
        X = []
        indices = []
        y = []

        for model in self.models:
            # grab n_samples_per for each model
            for ix in range(self.n_samples_per):
                model_index = self.models.index(model)
                key = self.keys[model][0]

                del self.keys[model][0]
                skeys = self.ifile[model][key].keys()

                # for each tree
                i = 0
                for skey in skeys:
                    X.append(np.array(self.ifile[model][key][skey]['x']))
                    indices.append(np.array(self.ifile[model][key][skey]['edge_index']) + 1)

                    y.append(model_index)

                    i += 1


        y = torch.LongTensor(np.hstack(y).astype(np.float32))

        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k])) for k in range(len(indices))])

        return batch, y

    def on_epoch_end(self):
        self.keys = {model: list(self.ifile[model].keys()) for model in self.models}
        for key in self.keys.keys():
            random.shuffle(self.keys[key])
