# -*- coding: utf-8 -*-
import numpy as np
from torch_geometric.data import Data, Batch, DataLoader
import torch
import random

import glob
import os
import copy

class TreeSeqGeneratorV2(object):
    def __init__(self, ifile, n_samples_per = 4, chunk_size = 5, models = "hard,hard-near,neutral,soft,soft-near"): # must be in order, see combine_h5s_v2
        self.ifile = ifile
        
        self.models = models.split(',')
        
        self.n_per = n_samples_per
        self.batch_size = chunk_size * self.n_per
        
        self.on_epoch_end()
        
    def on_epoch_end(self, shuffle = True):
        self.keys = sorted(list(self.ifile.keys()))
        if shuffle:
            random.shuffle(self.keys)
        
        self.ix = 0

    def __len__(self):
        return len(self.keys) // self.n_per
    
    def __getitem__(self, index):
        # features, edge_indices, and label, what sequence the graphs belong to
        X = []
        X1 = [] # tree-level features (same size as batch_)
        indices = []
        y = []
        batch_ = []
        
        for ix in range(self.n_per):
            key = self.keys[self.ix]
            
            x = np.array(self.ifile[key]['x'])
            y_ = np.array(self.ifile[key]['y'])
            x1 = np.array(self.ifile[key]['x1'])
            edge_index = np.array(self.ifile[key]['edge_index'])
            
            y.extend(y_)
            X.extend(list(x.reshape(x.shape[0] * x.shape[1], -1)))
            X1.extend(list(x1.reshape(x.shape[0] * x.shape[1], -1)))
            indices.extend(list(edge_index.reshape(x.shape[0] * x.shape[1], -1)))

        y = torch.LongTensor(np.array(y).astype(np.float32))
        X1 = torch.FloatTensor(np.array(X1))
    
        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k])) for k in range(len(indices))])

        return batch, X1, y

class TreeSeqGenerator(object):
    def __init__(self, ifile, models = None, means = 'intro_means.npz', n_samples_per = 16, 
                         sequence_length = 32, sequential = True, pad = False):
        if models is None:
            self.models = list(ifile.keys())
            print(self.models)
        else:
            self.models = models

        # hdf5 file we are reading from
        self.ifile = ifile
        means = np.load(means)
        
        self.pad = pad
        
        self.info_mean = means['v_mean'].reshape(1, -1)
        self.info_std = means['v_std'].reshape(1, -1)
        
        self.info_std[np.where(self.info_std == 0.)] = 1.

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
        
        # internal list for original sequence lengths observed
        self.lengths = []
        
        return
    
    # for internal use
    def get_single_model_batch(self, scattered_sample = False):
        Xs = []
        X1 = [] # tree-level features (same size as batch_)
        edge_index = []
        y = []
        
        for model in self.models:
            # features, edge_indices, and label, what sequence the graphs belong to
            
            while True:
                if self.counts[model] == len(self.keys[model]):
                    break
                
                key = self.keys[model][self.counts[model]]

                self.counts[model] += 1
                skeys = sorted(list(self.ifile[model][key].keys()))
                
                if not 'x' in skeys:
                    continue
                
                X_ = np.array(self.ifile[model][key]['x'])   
                if X_.shape[0] == 0:
                    continue
                
                if len(X_) > self.s_length:
                    if not scattered_sample:
                        ii = np.random.choice(range(len(X_) - self.s_length))
                        ii = range(ii, ii + self.s_length)
                    else:
                        ii = sorted(list(np.random.choice(range(len(X_)), self.s_length, replace = False)))
                    
                    X1_ = (np.array(self.ifile[model][key]['info']) - self.info_mean) / self.info_std
                    
                    break
                elif self.pad:
                    # pad out to this size
                    ii = list(range(self.s_length))
            
            # this guaruntees batches are always balanced
            if self.counts[model] == len(self.keys[model]):
                return None, None, None, None
            
            edges = np.array(self.ifile[model][key]['edge_index'], dtype = np.int32)
            # n_nodes, n_features
            s = (X_.shape[1], X_.shape[2])
            
            # record sequence length
            self.lengths.append(X_.shape[0])
            
            indices = []
            X = []
            
            for ii_ in ii:
                if ii_ < X_.shape[0]:
                    x = X_[ii_]
                    
                    ik = list(np.where(x[:,0] != 0))
                    x[ik,0] = np.log(x[ik,0])
                    
                    X.append(x)
                    
                    indices.append(edges[ii_])
                else:
                    x = np.zeros(s)
                    
                    X.append(x)
                    indices.append(None)
                
            X1.append(X1_[ii])
            X = np.array(X)
            Xs.append(X)
            
            y.append(model)
            edge_index.append(np.array(indices))
            
        return Xs, X1, edge_index, y

            
            
    
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
                    if X_.shape[0] == 0:
                        continue
                    
                    if len(X_) > self.s_length:
                        ii = np.random.choice(range(len(X_) - self.s_length))
                        ii = range(ii, ii + self.s_length)
                        
                        X1_ = (np.array(self.ifile[model][key]['info']) - self.info_mean) / self.info_std
                        
                        break
                    elif self.pad:
                        # pad out to this size
                        ii = list(range(self.s_length))
                
                # this guaruntees batches are always balanced
                if self.counts[model] == len(self.keys[model]):
                    return None, None, None, None
                
                edges = np.array(self.ifile[model][key]['edge_index'], dtype = np.int32)
                # n_nodes, n_features
                s = (X_.shape[1], X_.shape[2])
                
                for ii_ in ii:
                    if ii_ < X_.shape[0]:
                        x = X_[ii_]
                        
                        ik = list(np.where(x[:,0] != 0))
                        x[ik,0] = np.log(x[ik,0])
                        
                        X.append(x)
                        
                        indices.append(edges[ii_])
                    else:
                        x = np.zeros(s)
                        
                        X.append(x)
                        indices.append(None)
                    
                    batch_.extend(list(np.repeat(ij, x.shape[0])))
                    
                X1.append(X1_[ii])
                
                y.append(model_index)
                ij += 1

        if len(y) < 2:
            return None, None, None, None

        y = torch.LongTensor(np.hstack(y).astype(np.float32))
        X1 = torch.FloatTensor(np.array(X1))
    
        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k])) for k in range(len(indices))])

        return batch, y, X1, batch_
                
    def on_epoch_end(self):
        self.counts = dict()
        for model in self.models:
            self.counts[model] = 0

        for key in self.keys.keys():
            random.shuffle(self.keys[key])
            print(len(self.keys[key]))
            
    def __len__(self):
        return int(np.floor(np.min([len(self.keys[u]) for u in self.keys.keys()]) / self.n_samples_per))
    
def pad_matrix(x, new_size, axis = 0):
    # expects a genotype matrix (sites, n_individuals) shaped
    s = x.shape[0]
    
    if new_size > s:
        x_ = np.zeros((new_size - s, x.shape[1]))
        x = np.concatenate(x_)
    elif new_size < s:
        return None
    
    return x

class GenotypeMatrixGenerator(TreeSeqGenerator):
    def __init__(self, ifile, padded_size, **kwargs):
        super().__init__(ifile, **kwargs)
        
        self.padded_size = padded_size
        
    def __getitem__(self, index):
        X = []
        y = []
        
        for model in self.models:
            # grab n_samples_per for each model
            for ix in range(self.n_samples_per):
                while True:
                    if self.counts[model] == len(self.keys[model]):
                        break
                    
                    key = self.keys[model][self.counts[model]]
    
                    self.counts[model] += 1
                    skeys = sorted(list(self.ifile[model][key].keys()))
                    
                    if not 'x' in skeys:
                        continue
                    
                    X_ = np.array(self.ifile[model][key]['x_0'])
                    
                    X_ = pad_matrix(X_, self.padded_size)
                    if X_ is None:
                        continue
                    
                    break
                    
                X.append(X_)
                y.append(self.models.index(model))
                
            # guaruntees even class proportion batches
            if self.counts[model] == len(self.keys[model]):
                return None, None
                
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
            
        return X, y
                
        

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
