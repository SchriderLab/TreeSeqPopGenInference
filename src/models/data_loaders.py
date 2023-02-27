# -*- coding: utf-8 -*-
import numpy as np
from torch_geometric.data import Data, Batch, DataLoader
import torch
import random

import glob
import os
import copy

import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.spatial.distance import squareform

import zipfile
import PIL

import pyspng

class ManifoldNoise(object):
    def __init__(self, ifile, batch_size = 8, multiplier = 5.2619707157047735):
        x = np.load(ifile)
        self.mu = x['mu']
        self.sigma = x['sigma']
        
        n_grid_points = 4

        self.n = self.mu.shape[0]
        self.k = self.mu.shape[1]
        self.batch_size = batch_size
    
    def get_batch(self, n, ii = None):
        # standard normal
        z = np.random.normal(0, 1, (n, self.k))
        # sample the manifold uniformly
        if ii is None:
            ii = np.random.choice(range(self.n), n)
        
        # transform to sigma, mu
        #z *= self.sigma[ii]
        #z += self.mu[ii]
   
        return torch.FloatTensor(z), ii
    
    def get_batch_index(self, n, i):
        # standard normal
        z = np.random.normal(0, 1, (n, self.k))
        s = self.sigma[i].reshape(1, -1)
        mu = self.mu[i].reshape(1, -1)
        
        # transform to sigma, mu
        z *= s
        z += mu
        
        return torch.FloatTensor(z)

def chunks(lst, n):
    ret = dict()
    
    """Yield successive n-sized chunks from lst."""
    
    ix = 0
    for i in range(0, len(lst), n):
        ret[ix] = lst[i:i + n]

        ix += 1
        
    return ret

import cv2

class ImgGenerator(object):
    def __init__(self, path, batch_size = 8):
        self._path = path
        self.batch_size = batch_size

        
        self._zipfile = None

        if not os.path.isdir(self._path):
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
            
            PIL.Image.init()
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        else:
            self._type = 'dir'
            self._all_fnames = os.listdir(self._path)
            
            PIL.Image.init()
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
            
        self.data = chunks(self._image_fnames, 4096)
        
        return
    
    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()
    
    def _open_file(self, fname):
        if self._type == 'dir':
            return os.path.join(self._path, fname)
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None
    
    def get_batch(self, ii = None):
        if ii is None:
            ii = np.random.choice(range(len(self.data.keys())), self.batch_size)
        _ = [np.random.choice(self.data[u]) for u in ii]
        
        X = []
        for fname in _:
            f = self._open_file(fname)
            image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            
            if image.ndim == 2:
                image = image[:, :, np.newaxis] # HW => HWC
            image = image.transpose(2, 0, 1)
            
            X.append(image)
            
        X = torch.FloatTensor(np.array(X))
        
        return X

class ProjGenerator(object):
    def __init__(self, ifile, means, n_per = 8):
        self.n_per = n_per
        
        self.ifile = ifile
        means = np.load(means)
        
        self.x_mean = means['x_mean']
        self.x_std = means['x_std']
        
        self.x1_mean = means['x1_mean']
        self.x1_std = means['x1_std']
        
        self.x2_mean = means['x2_mean']
        self.x2_std = means['x2_std']
        
        
        
        self.counts = means['counts']
        
        self.on_epoch_end()
        
    def on_epoch_end(self, shuffle = True):
        self.keys = sorted(list(self.ifile.keys()))
        if shuffle:
            random.shuffle(self.keys)
        
        self.ix = 0
        
    def __len__(self):
        return len(self.keys) // self.n_per
        
    def __getitem__(self, index):
        X = []
        X1 = []
        X2 = []
        Y = []
        
        for ix in range(self.n_per):
            key = self.keys[self.ix]
            self.ix += 1
            
            x = torch.FloatTensor(np.array(self.ifile[key]['x']))
            x1 = torch.FloatTensor((np.array(self.ifile[key]['x1']) - self.x1_mean) / self.x1_std)
            
            x2 = np.array(self.ifile[key]['x2'])
            ii = np.where(self.x2_std != 0)
            x2[:,ii[0]] = (x2[:,ii[0]] - self.x2_mean[ii]) / self.x2_std[ii]
            x2 = torch.FloatTensor(x2)
            
            y = np.array(self.ifile[key]['y'])
            
            X.append(x)
            X1.append(x1)
            X2.append(x2)
            Y.extend(list(y))
            
        X = torch.cat(X, dim = 0)
        X1 = torch.cat(X1, dim = 0)
        X2 = torch.cat(X2, dim = 0)
        Y = torch.LongTensor(Y)
        
        return X, X1, X2, Y
            
class AutoGenerator(object):
    def __init__(self, ifile, means = 'seln_means.npz', D_means = './D_mean_stuff_v2/D_mean.npz', n_samples_per = 20, 
                 chunk_size = 5, models = "hard,hard-near,neutral,soft,soft-near"): # must be in order, see combine_h5s_v2
        self.ifile = ifile
        
        self.models = models.split(',')
        
        means = np.load(means)
        
        self.t_mean, self.t_std = tuple(means['times'])
        
        self.info_mean = means['v_mean'].reshape(1, 1, -1)
        self.info_std = means['v_std'].reshape(1, 1, -1)
        
        self.global_mean = means['v2_mean'].reshape(1, -1)
        self.global_std = means['v2_std'].reshape(1, -1)
        
        means = np.load(D_means)
        
        i,j = np.triu_indices(191)
        ii = list(np.where(i != j)[0])
        
        self.D_mean = means['mean']
        self.D_std = np.sqrt(means['var'])
        
        self.ii = ii
        
        self.info_std[np.where(self.info_std == 0.)] = 1.
        self.global_std[np.where(self.global_std == 0.)] = 1.
        self.D_std[np.where(self.D_std == 0.)] = 1.
        
        
        
        self.n_per = n_samples_per
        
        self.on_epoch_end()
        
    def on_epoch_end(self, shuffle = True):
        self.keys = sorted(list(self.ifile.keys()))
        if shuffle:
            random.shuffle(self.keys)
        
        self.ix = 0
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        
        if self.ix >= len(self.keys):
            return None, None
        
        # features, edge_indices, and label, what sequence the graphs belong to
        key = self.keys[self.ix]
        self.ix += 1
        
        # log scale and normalize times
        x = np.array(self.ifile[key]['x'])
        edge_index = np.array(self.ifile[key]['edge_index'])
        D = np.array(self.ifile[key]['D'])
        
        edge_index = edge_index.reshape(x.shape[0] * x.shape[1], 2, -1)
        D = D.reshape(x.shape[0] * x.shape[1], -1)[:,self.ii]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        
        ii = np.where(x[:,:,0] > 0) 
        x[ii[0],ii[1],0] = (np.log(x[ii[0],ii[1],0]) - self.t_mean) / self.t_std
        
        # log scale n_mutations
        ii = np.where(x[:,:,-1] > 0) 
        x[ii[0],ii[1],-1] = np.log(x[ii[0],ii[1],-1])
        
        ix = list(np.random.choice(range(x.shape[0]), self.n_per, replace = False))
        
        x = x[ix]
        edge_index = list(edge_index[ix])
        D = D[ix]
        
        i, j = np.where(D > 0)
        D[i, j] = np.log(D[i, j])

        Ds = []
        # mean normalize
        #D = (D - self.D_mean) / self.D_std
        for k in range(len(D)):
            #D[k] = (D[k] - np.min(D[k])) / (np.max(D[k]) - np.min(D[k]))
            
            Ds.append(squareform(D[k]))
        
        D = torch.FloatTensor(D)
        
        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(x[k]), edge_index=torch.LongTensor(edge_index[k])) for k in range(len(edge_index))])
        
        
        
        return batch, D
        

class TreeSeqGeneratorV2(object):
    def __init__(self, ifile, means = 'seln_means.npz', n_samples_per = 4, chunk_size = 5, models = "hard,hard-near,neutral,soft,soft-near"): # must be in order, see combine_h5s_v2
        self.ifile = ifile
        
        self.models = models.split(',')
        
        means = np.load(means)
        
        self.info_mean = means['v_mean'].reshape(1, 1, -1)
        self.info_std = means['v_std'].reshape(1, 1, -1)
        
        self.global_mean = means['v2_mean'].reshape(1, -1)
        self.global_std = means['v2_std'].reshape(1, -1)
        
        self.info_std[np.where(self.info_std == 0.)] = 1.
        self.global_std[np.where(self.global_std == 0.)] = 1.
        self.t_mean, self.t_std = tuple(means['times'])
        
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
        X2 = []
        batch_ = []
        
        for ix in range(self.n_per):
            key = self.keys[self.ix]
            self.ix += 1
            
            # log scale and normalize times
            x = np.array(self.ifile[key]['x'])
            ii = np.where(x[:,:,:,0] > 0) 
            x[ii[0],ii[1],ii[2],0] = (np.log(x[ii[0],ii[1],ii[2],0]) - self.t_mean) / self.t_std
            
            # log scale n_mutations
            ii = np.where(x[:,:,:,-1] > 0) 
            x[ii[0],ii[1],ii[2],-1] = np.log(x[ii[0],ii[1],ii[2],-1])
            
            y_ = np.array(self.ifile[key]['y'])
            x1 = (np.array(self.ifile[key]['x1']) - self.info_mean) / self.info_std
            edge_index = np.array(self.ifile[key]['edge_index'])
            mask = np.array(self.ifile[key]['mask'])
            global_vec = (np.array(self.ifile[key]['global_vec']) - self.global_mean) / self.global_std
            
            
            edge_index_ = []
            for k in range(x.shape[0]):
                _ = []
                
                e = edge_index[k]
                for j in range(e.shape[0]):
                    if mask[k][j] == 1:
                        _.append(torch.LongTensor(e[j]))
                    else:
                        _.append(torch.LongTensor([]))
                        
                edge_index_.extend(_)
                
                
            
            y.extend(y_)
            X.extend(list(x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])))
            X1.extend(list(x1))
            X2.extend(list(global_vec))
            indices.extend(edge_index_)

        y = torch.LongTensor(np.array(y).astype(np.float32))
        X1 = torch.FloatTensor(np.array(X1))
        X2 = torch.FloatTensor(np.array(X2))
    
        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [Data(x=torch.FloatTensor(X[k]), edge_index=indices[k]) for k in range(len(indices))])

        return batch, X1, X2, y

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
        try:
            means = np.load(means)
            
            self.pad = pad
            
            self.info_mean = means['v_mean'].reshape(1, -1)
            self.info_std = means['v_std'].reshape(1, -1)
            
            self.info_std[np.where(self.info_std == 0.)] = 1.
        except:
            pass

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
    def get_single_model_batch(self, n_samples = 3, sample_mode = 'scattered'):
        Xs = []
        Ds = [] # branch length distance matrices
        X1 = [] # tree-level features (same size as batch_)
        edge_index = []
        masks = []
        global_vec = []
        y = []
        
        for model in self.models:
            # features, edge_indices, and label, what sequence the graphs belong to
            
            for j in range(n_samples):
                while True:
                    if self.counts[model] == len(self.keys[model]):
                        break
                    
                    key = self.keys[model][self.counts[model]]
                    skeys = sorted(list(self.ifile[model][key].keys()))
                    
                    if not 'x' in skeys:
                        self.counts[model] += 1
                        continue
                    
                    X_ = np.array(self.ifile[model][key]['x']) 
                    X1_ = np.array(self.ifile[model][key]['info'])
                    D_ = np.array(self.ifile[model][key]['D'])[:,-2,:,:]
                    
                    u, v = np.triu_indices(D_.shape[1])
                    D_ = D_[:,u,v]
                    
                    if X_.shape[0] == 0:
                        self.counts[model] += 1
                        continue

                    if sample_mode == 'equi':
                        tree_bins = [0.] + list(np.cumsum(X1_[:,-2]))
                        #print(tree_bins)
                        ii = np.random.uniform(0., max(tree_bins), self.s_length)
                        
                        ii = sorted(list(np.digitize(ii, tree_bins) - 1))

                        padding = False
                        pad_size = (0, 0)
                    
                    break
                
                # this guaruntees batches are always balanced
                if self.counts[model] == len(self.keys[model]):
                    return None, None, None, None, None, None, None
                
                edges = np.array(self.ifile[model][key]['edge_index'], dtype = np.int32)
                global_vec_ = np.array(self.ifile[model][key]['global_vec'], dtype = np.float32)
                
                # n_nodes, n_features
                s = (X_.shape[1], X_.shape[2])
                
                # record sequence length
                self.lengths.append(X_.shape[0])
                
                indices = []
                mask = []
                
                X = []
                D = []
                
                for j in range(pad_size[0]):
                    x = np.zeros(s)
                    
                    X.append(x)
                    D.append(np.zeros(D_[0].shape))
                    indices.append(None)
                    mask.append(0.)
    
                for ii_ in ii:
                    x = X_[ii_]
                    d = D_[ii_]
                    
                    ik = list(np.where(x[:,0] > 0))
                    x[ik,0] = np.log(x[ik,0])
                    
                    X.append(x)
                    D.append(d)
                    mask.append(1.)
                    indices.append(edges[ii_])
                    
                X1.append(np.pad(X1_[ii], ((pad_size[0], pad_size[1]), (0, 0)), constant_values = 0.))
                    
                for j in range(pad_size[1]):
                    x = np.zeros(s)
                    
                    X.append(x)
                    D.append(np.zeros(D_[0].shape))
                    indices.append(None)
                    mask.append(0.)
                    
                X = np.array(X)
                D = np.array(D)
                
                Xs.append(X)
                Ds.append(D)
                
                y.append(model)
                s = [u for u in indices if u is not None][-1].shape
                
                for j in range(len(indices)):
                    if indices[j] is None:
                        indices[j] = np.zeros(s, dtype = np.int32)
                
                
                mask = np.array(mask, dtype = np.uint8)
                edge_index.append(np.array(indices))
                masks.append(mask)
                global_vec.append(global_vec_)
                
                if padding or self.lengths[-1] < self.s_length:
                    break
            
            self.counts[model] += 1
            
            
        return Xs, X1, edge_index, masks, global_vec, y, Ds

            
            
    
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
        x = np.concatenate([x,x_],axis=0)
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
                    
                    X_ = np.array(self.ifile[model][key]['seriated_real'])
                    
                    X_ = pad_matrix(np.transpose(X_), self.padded_size)
                    X_ = np.transpose(X_,(1,0))
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
