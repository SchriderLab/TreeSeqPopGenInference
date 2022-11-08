import numpy as np
import h5py
import multiprocessing as mp
from tqdm import tqdm
from seriate_file import seriate
from scipy.spatial.distance import pdist,squareform,cdist
from scipy.optimize import linear_sum_assignment 


def seriate_x(x, metric = 'cosine'):
    Dx = pdist(x, metric = metric)
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx, timeout = 0)

    return x[ix], ix

def pad_matrix(x, new_size, axis = 0):
    # expects a genotype matrix (sites, n_individuals) shaped
    s = x.shape[0]
    
    if new_size > s:
        x_ = np.zeros((new_size - s, x.shape[1]))
        x = np.concatenate([x,x_],axis=0)
    elif new_size < s:
        return None
    
    return x

def seriate_(X_):
    X_pop1 = X_[:20,:] #first species
    X_pop2 = X_[20:,:] #second species

    X_sorted_pop1,X_sorted_indicies = seriate_x(X_pop1)
    X_sorted_pop2,X_sorted_indicies = seriate_x(X_pop2)     

    X_sorted = np.concatenate((X_sorted_pop1,X_sorted_pop2))
    return X_sorted

def seriate_match(X_):
    X_pop1 = X_[:20,:] #first species
    X_pop2 = X_[20:,:] #second species

    X_pop1_padded = pad_matrix(X_pop1,32)
    X_pop2_padded = pad_matrix(X_pop2,32)

    X_pop1_sorted,X_sorted_indicies = seriate_x(X_pop1_padded)
    
    dx =  cdist(X_pop1_padded,X_pop2_padded,metric = 'cosine')
    dx[np.where(np.isnan(dx))] = 0.

    row_ind,col_ind = linear_sum_assignment(dx)

    X_pop2_matched = X_pop2_padded[col_ind]

    X_2 = np.concatenate((X_pop1_sorted,X_pop2_matched))
    return np.array([X_pop1_sorted,X_pop2_matched])


def worker(args):
    sample, idx = args #Unpack the packed tuples
    return seriate_match(sample), idx

hf = h5py.File('/pine/scr/l/o/lobanov/src/models/intro_trees_val.hdf5', 'r+')
hf_out = h5py.File('/pine/scr/l/o/lobanov/src/models/seriated_intro_trees_val.hdf5', 'w')
pool = mp.Pool(mp.cpu_count()) #Max number of CPUs given to program (or an arbitrary int if desired)

for model in list(hf.keys()): #Iterate through models
    print(model)
    grp = hf_out.create_group(model)
    for i in tqdm(range(1,len(hf[model])-100,100), desc=f"Seriating data in {model}"): #Only ranging for minimum length since keys have different length for parallel use.
        batch = [] #Tuples of (sample, id)
        for j in range(i, i+100): #Iterate through each idx in batch and store with raw data
            batch.append((np.array(hf[model][str(j)]['x_0']), j))
        results = pool.map(worker, batch, chunksize=4) #Now we can throw into parallel processing, the most time intensive part
        for res in results:
            subgrp = grp.create_group(str(res[1]))
            sub_subgrp = subgrp.create_dataset('seriated_matched',data=res[0])
    
    if round(len(hf[model]),-2) >= len(hf[model]):
        num_left = round(len(hf[model]),-2) - 99
    else:
        num_left = round(len(hf[model]),-2) + 1 
    print(num_left)

    for i in tqdm(range(num_left, len(hf[model])), desc=f"Seriating data in"):  #for last 100 samples that cant be pooled
        subgrp = grp.create_group(str(i))
        X_ = np.array(hf[model][str(i)]['x_0'])
        results = seriate_match(X_)
        sub_subgrp = subgrp.create_dataset('seriated_matched',data=results)  

    subgrp = grp.create_group('0') #for first sample to work with range function in pooled version
    X_ = hf[model]['0']['x_0']
    results = seriate_match(X_)
    sub_subgrp = subgrp.create_dataset('seriated_matched',data=results)

pool.close()
pool.join()
hf.close()

#'mig12' seriated up to index 143200, needs 143267
#'mig21' seriated up to index 128700, needs 128752
#'noMig' seriated up to index 143900, needs 144000