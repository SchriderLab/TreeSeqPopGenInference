# -*- coding: utf-8 -*-
import numpy as np
from torch_geometric.data import Data, Batch, DataLoader
import torch
import random

import time
import logging

def chunks(lst, n):
    ret = dict()

    """Yield successive n-sized chunks from lst."""

    ix = 0
    for i in range(0, len(lst), n):
        ret[ix] = lst[i : i + n]

        ix += 1

    return ret

class TreeSeqGeneratorV3(object):
    def __init__(
        self,
        ifile,
        means="demography_means.npz",
        regression=False,
        n_samples_per=4,
        models="none",
        log_y = True,
        y_ix = None,
        return_params = False
    ):  # must be in order, see combine_h5s_v2
        self.ifile = ifile

        self.y_ix = y_ix

        self.models = models.split(",")
        self.log_y = log_y

        means = np.load(means)
        

        self.info_mean = means["v_mean"].reshape(1, 1, -1)
        self.info_std = means["v_std"].reshape(1, 1, -1)

        self.global_mean = means["v2_mean"].reshape(1, -1)
        self.global_std = means["v2_std"].reshape(1, -1)

        self.return_params = return_params
        self.regression = regression

        if self.regression:
            self.y_mean = means["y_mean"].reshape(1, -1)
            self.y_std = means["y_std"].reshape(1, -1)

        self.info_std[np.where(self.info_std == 0.0)] = 1.0
        self.global_std[np.where(self.global_std == 0.0)] = 1.0
        self.t_mean, self.t_std = tuple(means["times"])

        self.n_per = n_samples_per

        self.on_epoch_end()
        
        x = self.ifile[self.keys[0]]['x']
        
        self.chunk_size = x.shape[0]
        self.batch_size = self.chunk_size * self.n_per
         

    def on_epoch_end(self, shuffle=True):
        self.keys = sorted(list(self.ifile.keys()))
        if shuffle:
            random.shuffle(self.keys)

        self.ix = 0

    def __len__(self):
        return len(self.keys) // self.n_per

    def __getitem__(self, index):
        # features, edge_indices, and label, what sequence the graphs belong to
        X = []
        X1 = []  # tree-level features (same size as batch_)
        indices = []
        y = []
        X2 = []
        params = []
        
        lengths = []
        
        batch_ = []
        mask_indices = []
        
        batch_ii = 0

        t0 = time.time()
        for ix in range(self.n_per):
            if self.ix > len(self.keys) - 1:
                break
            
            key = self.keys[self.ix]
            self.ix += 1

            # log scale and normalize times
            x = np.array(self.ifile[key]["x"])
            ii = np.where(x[:, :, :, 0] > 0)
            x[ii[0], ii[1], ii[2], 0] = (
                np.log(x[ii[0], ii[1], ii[2], 0]) - self.t_mean
            ) / self.t_std

            # log scale n_mutations
            ii = np.where(x[:, :, :, -1] > 0)
            x[ii[0], ii[1], ii[2], -1] = np.log(x[ii[0], ii[1], ii[2], -1])

            y_ = np.array(self.ifile[key]["y"])
            if self.y_ix is not None:
                y_ = y_[:,[self.y_ix]]
            
            # normalize the tree summary vectors
            x1 = (np.array(self.ifile[key]["x1"]) - self.info_mean) / self.info_std
            edge_index = np.array(self.ifile[key]["edge_index"])

            mask = np.array(self.ifile[key]["mask"])
            mask_indices.extend(np.concatenate([np.expand_dims(u, 0) for u in np.where(mask == 1)]).T + np.array([[batch_ii, 0]]))
            
            # normalize the tree-seq summary vector
            global_vec = (
                np.array(self.ifile[key]["global_vec"]) - self.global_mean
            ) / self.global_std

            edge_index_ = []
            
            # write down the indices of which graph belongs to which 
            # example (not sure if we use this anymore...)
            ii = 0
            for k in range(mask.shape[0]):
                n_graphs = np.sum(mask[k])
                
                batch_.extend(np.ones(n_graphs, dtype = np.int32) * batch_ii)
                batch_ii += 1
            
            # 128 graphs
            # number of graphs in sequence
            # some graphs are just 0s (we assume it's a padded sequence), but we put them in Batch object
            # with an empty list of edges
    
            # for each graph sequence
            for k in range(x.shape[0]):
                _ = []

                e = edge_index[k]

                # for each graph
                for j in range(e.shape[0]):
                    # is it masked?
                    if mask[k][j] == 1:
                        # remove the root node
                        e_ = np.prod(e[j], axis=0)
                        ii = np.where(e_ >= 0)[0]
                    
                        _.append(torch.LongTensor(e[j][:, ii]))

                
                x_ = x[k][np.where(mask[k] == 1)[0]]
                
                lengths.append(x_.shape[0])
                
                X.append(x_)
                y.append(np.expand_dims(y_[k], 0))
            
                X1.append(x1[k][np.where(mask[k] == 1)[0]])
                X2.append(np.expand_dims(global_vec[k], 0))
                
                
                indices.append(torch.cat([torch.unsqueeze(u,0) for u in _]))
            
            if self.return_params:
                params.append(self.ifile[key]['params'])

        if len(X) == 0:
            if not self.return_params:
                return None, None, None, None
            else:
                return None, None, None, None, None

        # sort the batch in length descending (required for torchs pack and pad operations for the GRU)
        ii = np.argsort(lengths)[::-1]
        
        # sort all the lists
        X = [X[u] for u in ii]
        y = [y[u] for u in ii]
        X1 = [X1[u] for u in ii]
        X2 = [X2[u] for u in ii]
        indices = [indices[u] for u in ii]
        
        # cat to arrays
        X = np.concatenate(X, 0)
        y = np.concatenate(y, 0)
        X1 = np.concatenate(X1, 0)
        X2 = np.concatenate(X2, 0)
        indices = torch.cat(indices, 0)
        if self.return_params:
            params = np.concatenate(params, 0)

        ii = 0
        
        batch_indices = []
        for l in sorted(lengths, reverse = True):
            batch_indices.append(torch.LongTensor(np.array(range(ii, ii + l))))
            ii += l
            
        if self.regression:
            if self.log_y:
                y = np.log(np.array(y).astype(np.float32))
            else:
                y = np.array(y, dtype = np.float32)
            y = torch.FloatTensor(
                (y - self.y_mean) / self.y_std
            )
        else:
            y = torch.LongTensor(np.array(y))
        
        X1 = torch.FloatTensor(np.array(X1))
        X2 = torch.FloatTensor(np.array(X2))

        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [
                Data(x=torch.FloatTensor(X[k]), edge_index=indices[k])
                for k in range(len(indices))
            ]
        )
        
        batch.rep_indices = torch.LongTensor(np.array(batch_))
        batch.mask_indices = torch.LongTensor(np.array(mask_indices, dtype = np.int32))
        
        batch.batch_indices = batch_indices

        logging.debug('clocked at {} s'.format(time.time() - t0))
        if self.return_params:
            return batch, X1, X2, y, params
        else:        
            return batch, X1, X2, y

class TreeSeqGeneratorV2(object):
    def __init__(
        self,
        ifile,
        means="demography_means.npz",
        regression=False,
        n_samples_per=4,
        chunk_size=1,
        models="none",
        log_y = True,
        y_ix = None
    ):  # must be in order, see combine_h5s_v2
        self.ifile = ifile

        self.y_ix = y_ix

        self.models = models.split(",")
        self.log_y = log_y

        means = np.load(means)

        self.info_mean = means["v_mean"].reshape(1, 1, -1)
        self.info_std = means["v_std"].reshape(1, 1, -1)

        self.global_mean = means["v2_mean"].reshape(1, -1)
        self.global_std = means["v2_std"].reshape(1, -1)

        self.regression = regression

        if self.regression:
            self.y_mean = means["y_mean"].reshape(1, -1)
            self.y_std = means["y_std"].reshape(1, -1)

        self.info_std[np.where(self.info_std == 0.0)] = 1.0
        self.global_std[np.where(self.global_std == 0.0)] = 1.0
        self.t_mean, self.t_std = tuple(means["times"])

        self.n_per = n_samples_per
        self.batch_size = chunk_size * self.n_per

        self.on_epoch_end()

    def on_epoch_end(self, shuffle=True):
        self.keys = sorted(list(self.ifile.keys()))
        if shuffle:
            random.shuffle(self.keys)

        self.ix = 0

    def __len__(self):
        return len(self.keys) // self.n_per

    def __getitem__(self, index):
        # features, edge_indices, and label, what sequence the graphs belong to
        X = []
        X1 = []  # tree-level features (same size as batch_)
        indices = []
        y = []
        X2 = []
        batch_ = []

        t0 = time.time()
        for ix in range(self.n_per):
            if self.ix > len(self.keys) - 1:
                break
            
            key = self.keys[self.ix]
            self.ix += 1

            # log scale and normalize times
            x = np.array(self.ifile[key]["x"])
            ii = np.where(x[:, :, :, 0] > 0)
            x[ii[0], ii[1], ii[2], 0] = (
                np.log(x[ii[0], ii[1], ii[2], 0]) - self.t_mean
            ) / self.t_std

            # log scale n_mutations
            ii = np.where(x[:, :, :, -1] > 0)
            x[ii[0], ii[1], ii[2], -1] = np.log(x[ii[0], ii[1], ii[2], -1])

            y_ = np.array(self.ifile[key]["y"])
            if self.y_ix is not None:
                y_ = y_[:,[self.y_ix]]
            
            x1 = (np.array(self.ifile[key]["x1"]) - self.info_mean) / self.info_std
            edge_index = np.array(self.ifile[key]["edge_index"])

            mask = np.array(self.ifile[key]["mask"])
            global_vec = (
                np.array(self.ifile[key]["global_vec"]) - self.global_mean
            ) / self.global_std

            edge_index_ = []
            # number of graphs in sequence
            # some graphs are just 0s (we assume it's a padded sequence), but we put them in Batch object
            # with an empty list of edges
            for k in range(x.shape[0]):
                _ = []

                e = edge_index[k]
                for j in range(e.shape[0]):
                    if mask[k][j] == 1:
                        e_ = np.product(e[j], axis=0)
                        ii = np.where(e_ >= 0)[0]
                        _.append(torch.LongTensor(e[j][:, ii]))
                    else:
                        _.append(torch.LongTensor([]))

                edge_index_.extend(_)

            y.extend(y_)
            X.extend(list(x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])))

            X1.extend(list(x1))
            X2.extend(list(global_vec))
            indices.extend(edge_index_)

        if len(X) == 0:
            return None, None, None, None
        
        if self.regression:
            if self.log_y:
                y = np.log(np.array(y).astype(np.float32))
            else:
                y = np.array(y, dtype = np.float32)
            y = torch.FloatTensor(
                (y - self.y_mean) / self.y_std
            )
        else:
            y = torch.LongTensor(np.array(y))
        
        X1 = torch.FloatTensor(np.array(X1))
        X2 = torch.FloatTensor(np.array(X2))

        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [
                Data(x=torch.FloatTensor(X[k]), edge_index=indices[k])
                for k in range(len(indices))
            ]
        )

        logging.debug('clocked at {} s'.format(time.time() - t0))
        return batch, X1, X2, y

class TreeSeqGenerator(object):
    def __init__(
        self,
        ifile,
        models=None,
        means="intro_means.npz",
        n_samples_per=16,
        sequence_length=32,
        sequential=True,
        pad=False,
        categorical=True,
    ):
        if models is None:
            self.models = list(ifile.keys())
            print(self.models)
        else:
            self.models = models

        self.categorical = categorical

        # hdf5 file we are reading from
        self.ifile = ifile
        try:
            means = np.load(means)

            self.pad = pad

            self.info_mean = means["v_mean"].reshape(1, -1)
            self.info_std = means["v_std"].reshape(1, -1)

            self.info_std[np.where(self.info_std == 0.0)] = 1.0
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
    
    # takes a single tree sequence and normalizes it / prepares it for inference
    def prepare_seq(self, x, x1, edge_index, mask, global_vec, y):
        return
    
    def get_seq(self, model, key, sample_mode = "sequential"):
        X_ = np.array(self.ifile[model][key]["x"])
        X1_ = np.array(self.ifile[model][key]["info"])

        if X_.shape[0] == 0:
            return None, None, None, None, None, None
        
        if sample_mode == "equi":
            tree_bins = [0.0] + list(np.cumsum(X1_[:, -2]))
            ii = np.random.uniform(0.0, max(tree_bins), self.s_length)

            ii = sorted(list(np.digitize(ii, tree_bins) - 1))
            
            padding = False
            pad_size = (0, 0)
        elif sample_mode == "sequential":
            if X_.shape[0] == self.s_length:
                ii = range(X_.shape[0])
                padding = False
                pad_size = (0, 0)
            elif X_.shape[0] > self.s_length:
                ii = np.random.choice(range(X_.shape[0] - self.s_length))
                ii = range(ii, ii + self.s_length)
                
                padding = False
                pad_size = (0, 0)
            else:
                to_pad = self.s_length - X_.shape[0]
                if to_pad % 2 == 0:
                    pad_size = (to_pad // 2, to_pad // 2)
                else:
                    pad_size = (to_pad // 2, to_pad // 2 + 1)
                    
                ii = range(X_.shape[0])
                padding = True

        edges = np.array(self.ifile[model][key]["edge_index"], dtype=np.int32)
        global_vec = np.array(
            self.ifile[model][key]["global_vec"], dtype=np.float32
        )

        # n_nodes, n_features
        s = (X_.shape[1], X_.shape[2])

        # record sequence length
        self.lengths.append(X_.shape[0])

        indices = []
        mask = []

        X = []

        for j in range(pad_size[0]):
            x = np.zeros(s)

            X.append(x)
            indices.append(None)
            mask.append(0.0)

        for ii_ in ii:
            x = X_[ii_]

            ik = list(np.where(x[:, 0] > 0))
            x[ik, 0] = np.log(x[ik, 0])

            X.append(x)
            mask.append(1.0)
            indices.append(edges[ii_])

        X1 = np.pad(
                X1_[ii],
                ((pad_size[0], pad_size[1]), (0, 0)),
                constant_values=0.0,
            )
        
        for j in range(pad_size[1]):
            x = np.zeros(s)

            X.append(x)
            indices.append(None)
            mask.append(0.0)

        X = np.array(X)
        
        s = [u for u in indices if u is not None][-1].shape

        for j in range(len(indices)):
            if indices[j] is None:
                indices[j] = np.zeros(s, dtype=np.int32)

        mask = np.array(mask, dtype=np.uint8)
        
        s = [u for u in indices if u is not None][-1].shape

        for j in range(len(indices)):
            if indices[j] is None:
                indices[j] = np.zeros(s, dtype=np.int32)
        
        edge_index = np.array(indices)
    
        if self.categorical:
            y = model
        else:
            try:
                y = np.array(self.ifile[model][key]["y"], dtype=np.float32)
            except:
                return None, None, None, None, None, None
    
        return X, X1, edge_index, mask, global_vec, y

    # for internal use
    def get_single_model_batch(self, n_samples=1, sample_mode="scattered"):
        Xs = []
        Ds = []  # branch length distance matrices
        X1 = []  # tree-level features (same size as batch_)
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

                    if not "x" in skeys:
                        self.counts[model] += 1
                        continue

                    X_ = np.array(self.ifile[model][key]["x"])
                    X1_ = np.array(self.ifile[model][key]["info"])

                    if X_.shape[0] == 0:
                        self.counts[model] += 1
                        continue

                    if sample_mode == "equi":
                        tree_bins = [0.0] + list(np.cumsum(X1_[:, -2]))
                        ii = np.random.uniform(0.0, max(tree_bins), self.s_length)

                        ii = sorted(list(np.digitize(ii, tree_bins) - 1))
                        
                        padding = False
                        pad_size = (0, 0)
                    elif sample_mode == "sequential":
                        if X_.shape[0] == self.s_length:
                            ii = range(X_.shape[0])
                            padding = False
                            pad_size = (0, 0)
                        elif X_.shape[0] > self.s_length:
                            ii = np.random.choice(range(X_.shape[0] - self.s_length))
                            ii = range(ii, ii + self.s_length)
                            
                            padding = False
                            pad_size = (0, 0)
                        else:
                            to_pad = self.s_length - X_.shape[0]
                            if to_pad % 2 == 0:
                                pad_size = (to_pad // 2, to_pad // 2)
                            else:
                                pad_size = (to_pad // 2, to_pad // 2 + 1)
                                
                            ii = range(X_.shape[0])
                            padding = True
                    break

                # this guaruntees batches are always balanced
                if self.counts[model] == len(self.keys[model]):
                    return None, None, None, None, None, None

                edges = np.array(self.ifile[model][key]["edge_index"], dtype=np.int32)
                global_vec_ = np.array(
                    self.ifile[model][key]["global_vec"], dtype=np.float32
                )

                # n_nodes, n_features
                s = (X_.shape[1], X_.shape[2])

                # record sequence length
                self.lengths.append(X_.shape[0])

                indices = []
                mask = []

                X = []

                for j in range(pad_size[0]):
                    x = np.zeros(s)

                    X.append(x)
                    indices.append(None)
                    mask.append(0.0)

                for ii_ in ii:
                    x = X_[ii_]

                    ik = list(np.where(x[:, 0] > 0))
                    x[ik, 0] = np.log(x[ik, 0])

                    X.append(x)
                    mask.append(1.0)
                    indices.append(edges[ii_])

                X1.append(
                    np.pad(
                        X1_[ii],
                        ((pad_size[0], pad_size[1]), (0, 0)),
                        constant_values=0.0,
                    )
                )

                for j in range(pad_size[1]):
                    x = np.zeros(s)

                    X.append(x)
                    indices.append(None)
                    mask.append(0.0)

                X = np.array(X)

                Xs.append(X)

                if self.categorical:
                    y.append(model)
                else:
                    try:
                        y_ = np.array(self.ifile[model][key]["y"], dtype=np.float32)
                    except:
                        return None, None, None, None, None, None
                    y.append(y_)

                s = [u for u in indices if u is not None][-1].shape

                for j in range(len(indices)):
                    if indices[j] is None:
                        indices[j] = np.zeros(s, dtype=np.int32)

                mask = np.array(mask, dtype=np.uint8)
                edge_index.append(np.array(indices))
                masks.append(mask)
                global_vec.append(global_vec_)

                if padding or self.lengths[-1] < self.s_length:
                    break

            self.counts[model] += 1

        return Xs, X1, edge_index, masks, global_vec, y

    def __getitem__(self, index):
        # features, edge_indices, and label, what sequence the graphs belong to
        X = []
        X1 = []  # tree-level features (same size as batch_)
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

                    if not "x" in skeys:
                        continue

                    X_ = np.array(self.ifile[model][key]["x"])
                    if X_.shape[0] == 0:
                        continue

                    if len(X_) > self.s_length:
                        ii = np.random.choice(range(len(X_) - self.s_length))
                        ii = range(ii, ii + self.s_length)

                        X1_ = (
                            np.array(self.ifile[model][key]["info"]) - self.info_mean
                        ) / self.info_std

                        break
                    elif self.pad:
                        # pad out to this size
                        ii = list(range(self.s_length))

                # this guaruntees batches are always balanced
                if self.counts[model] == len(self.keys[model]):
                    return None, None, None, None

                edges = np.array(self.ifile[model][key]["edge_index"], dtype=np.int32)
                # n_nodes, n_features
                s = (X_.shape[1], X_.shape[2])

                for ii_ in ii:
                    if ii_ < X_.shape[0]:
                        x = X_[ii_]

                        ik = list(np.where(x[:, 0] != 0))
                        x[ik, 0] = np.log(x[ik, 0])

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
            [
                Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k]))
                for k in range(len(indices))
            ]
        )

        return batch, y, X1, batch_

    def on_epoch_end(self):
        self.counts = dict()
        for model in self.models:
            self.counts[model] = 0

        for key in self.keys.keys():
            random.shuffle(self.keys[key])

    def __len__(self):
        return int(
            np.floor(
                np.min([len(self.keys[u]) for u in self.keys.keys()])
                / self.n_samples_per
            )
        )


def pad_matrix(x, new_size, axis=0):
    # expects a genotype matrix (sites, n_individuals) shaped
    s = x.shape[0]

    if new_size > s:
        x_ = np.zeros((new_size - s, x.shape[1]))
        x = np.concatenate([x, x_], axis=0)
    elif new_size < s:
        return None

    return x


class GenotypeMatrixGenerator(object):
    def __init__(self, ifile, models):
        self.ifile = ifile
        if models is None:
            self.models = list(ifile.keys())
            print(self.models)
        else:
            self.models = models

    def __len__(self):
        return int(list(self.ifile[self.models[0]].keys())[-1]) + 1

    def __getitem__(self, index):
        X = []
        y = []

        for idx, model in enumerate(self.models):
            x_arr = np.array(self.ifile[f"{model}/{index}/x"])
            X.append(x_arr)
            y.extend([idx] * x_arr.shape[0])

        X = np.concatenate(X)
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        

        return X, y


class DemoGenotypeMatrixGenerator(object):
    def __init__(self, ifile, models):
        super().__init__()
        self.ifile = ifile
        self.models = models

    def __getitem__(self, index):
        X = []
        y = []

        for idx, model in enumerate(self.models):
            x_arr = np.array(self.ifile[f"demo/{index}/x_0"])
            X.append(x_arr)
            y.extend([idx] * x_arr.shape[0])

        y = np.array(y)
        if self.y_ix is not None:
            y = y[:,[self.y_ix]]

        X = np.concatenate(X)
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        return X, y

import h5py

class GenomatClassGenerator(object):
    def __init__(self, ifile, batch_size = 2):
        self.ifile = h5py.File(ifile, 'r')
        self.classes = sorted(list(self.ifile.keys()))
        
        print(self.classes)
        
        keys = dict()
        
        for c in self.classes:
            keys[c] = list(self.ifile[c].keys())
            
        self.keys = keys
        
        self.l = min([len(self.keys[u]) for u in self.classes]) // batch_size
        
        self.batch_size = batch_size        
        
        self.on_epoch_end()
        
    def on_epoch_end(self):
        for c in self.classes:
            random.shuffle(self.keys[c])
            
        self.ix = 0
        
    def __len__(self):
        return self.l
    
    def __getitem__(self, index):
        X = []
        y = []
        
        for k in range(self.batch_size):
            if any([self.ix >= len(self.keys[c]) for c in self.classes]):
                break
            
            for ix, c in enumerate(self.classes):
                
                key = self.keys[c][self.ix]
                
                x = np.array(self.ifile[c][key]['x'])
                
                X.extend(x)
                y.extend([ix for u in range(x.shape[0])])
                
            self.ix += 1
        
        if len(X) == 0:
            return None, None
            
        X = torch.FloatTensor(np.array(X))
        
        if len(X.shape) == 3:
            X = torch.unsqueeze(X, 1)
        
        y = torch.LongTensor(np.array(y))
            
        return X, y

class GenomatGenerator(object):
    def __init__(self, ifile, means, y_ix = None, batch_size = 4, classification = False, log_y = False):
        self.ifile = h5py.File(ifile, 'r')
        self.keys = list(self.ifile.keys())
        
        self.batch_size = batch_size
        self.y_ix = y_ix

        means = np.load(means)
        self.y_mean = means['y_mean'].reshape(1, -1)
        self.y_std = means['y_std'].reshape(1, -1)
        
        self.log_y = log_y
        
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.keys) // self.batch_size
        
    def __getitem__(self, index):
        X = []
        y = []
        
        for ix in range(index * self.batch_size, (index + 1) * self.batch_size):
            if ix >= len(self.keys):
                break
            
            key = self.keys[ix]
            X.extend(np.array(self.ifile[key]['x']))
            y.extend(np.array(self.ifile[key]['y']))

        if len(X) == 0:
            return None, None
            
        y = np.array(y)
        y = np.squeeze(y)
        if self.log_y:
            y = np.log(y)
            
        X = np.array(X)
        if self.y_ix is not None:
            y = np.array(y)[:,[self.y_ix]]
            
        y = (y - self.y_mean) / self.y_std
            
        # expand for the channel dimension
        if len(X.shape) == 3:
            X = np.expand_dims(X, 1)
            
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        return X, y

    def on_epoch_end(self):
        random.shuffle(self.keys)
        

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
        return int(
            np.floor(
                np.min([len(self.keys[u]) for u in self.keys.keys()])
                / self.n_samples_per
            )
        )

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
                    X.append(np.array(self.ifile[model][key][skey]["x"]))
                    indices.append(
                        np.array(self.ifile[model][key][skey]["edge_index"]) + 1
                    )

                    y.append(model_index)

                    i += 1

        y = torch.LongTensor(np.hstack(y).astype(np.float32))

        # use PyTorch Geometrics batch object to make one big graph
        batch = Batch.from_data_list(
            [
                Data(x=torch.FloatTensor(X[k]), edge_index=torch.LongTensor(indices[k]))
                for k in range(len(indices))
            ]
        )

        return batch, y

    def on_epoch_end(self):
        self.keys = {model: list(self.ifile[model].keys()) for model in self.models}
        for key in self.keys.keys():
            random.shuffle(self.keys[key])
