# -*- coding: utf-8 -*-

import gzip
import numpy as np

def split(word):
    return [char for char in word]

######
# generic function for msmodified
# ----------------
def load_data(msFile, ancFile, n = None, leave_out_last = False):
    msFile = gzip.open(msFile, 'r')

    # no migration case
    try:
        ancFile = gzip.open(ancFile, 'r')
    except:
        ancFile = None

    ms_lines = [u.decode('utf-8') for u in msFile.readlines()]

    if leave_out_last:
        ms_lines = ms_lines[:-1]

    if ancFile is not None:
        idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]
    else:
        idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]
        
            
    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1] += ['\n']

    if ancFile is not None:
        anc_lines = [u.decode('utf-8') for u in ancFile.readlines()]
    else:
        anc_lines = None
        
    X = []
    Y = []
    P = []
    intros = []
    for chunk in ms_chunks:
        line = chunk[0]
        if '*' in line:
            intros.append(True)
        else:
            intros.append(False)
        
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)
        
        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]], dtype = np.uint8)
        # destroy the perfect information regarding
        # which allele is the ancestral one
        for k in range(x.shape[1]):
            if np.sum(x[:,k]) > x.shape[0] / 2.:
                x[:,k] = 1 - x[:,k]
            elif np.sum(x[:,k]) == x.shape[0] / 2.:
                if np.random.choice([0, 1]) == 0:
                    x[:,k] = 1 - x[:,k]
        
        if anc_lines is not None:
            y = np.array([list(map(int, split(u.replace('\n', '')))) for u in anc_lines[:len(pos)]], dtype = np.uint8)
            y = y.T
            
            del anc_lines[:len(pos)]
        else:
            y = np.zeros(x.shape, dtype = np.uint8)
            
        print(pos.shape, x.shape)
        assert len(pos) == x.shape[1]
        
        if n is not None:
            x = x[:n,:]
            y = y[:n,:]
            
        X.append(x)
        Y.append(y)
        P.append(pos)
        
    return X, Y, P, intros