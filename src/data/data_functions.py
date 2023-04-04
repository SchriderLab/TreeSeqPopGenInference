# -*- coding: utf-8 -*-

import gzip
import numpy as np
import networkx as nx
import itertools

from io import StringIO
from skbio import read
from skbio.tree import TreeNode
import re

# writes a space separated .tbs file (text)
# which contains the demographic parameters for an ms two-population simulation
def writeTbsFile(params, outFileName):
    with open(outFileName, "w") as outFile:
        for paramVec in params:
            outFile.write(" ".join([str(x) for x in paramVec]) + "\n")

def read_ms_tree(ifile, n = 34, L = 10000):
    ifile = gzip.open(ifile, 'r')
    
    ms_lines = ifile.readlines()
    ms_lines = [u.decode('utf-8') for u in ms_lines]
    
    idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]
    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1].append('\n')
    
    ret = dict()
    
    # edges in the trees
    ret['edges'] = []
    # preorder traversal index lists
    ret['order'] = []
    # position
    ret['positions'] = []
    # ages of the nodes, back in time
    ret['ages'] = []
    # alignment matrix (whole simulation)
    ret['alignment'] = []
    for chunk in ms_chunks:
        c = chunk[1:-1]

        c = [u for u in c if '[' in u]
        c = [u.replace('\n','') for u in c]
        
        align_lines = chunk[-(n + 1):-1]
        pos_line = [u for u in chunk if 'positions:' in u][0].replace('\n', '')

        pos_ = np.round(np.array(list(map(float, pos_line.split(' ')[1:-1]))) * L).astype(np.int32)
        
        align_lines = [u.replace('\n','') for u in align_lines]

        x = [np.array(list(map(int, [u for u in l])), dtype = np.uint8) for l in align_lines]
        x = np.array(x, dtype = np.float32).T
        
        ret['alignment'].append(x)
    
        e = []
        orders = []
        ages = []
        positions = []
        
        ls = []
        
        pos = 0
        for s in c:
            f = StringIO(s)  
            t = read(f, format="newick", into=TreeNode)
            
            # get the position of the tree (cumulative sum of the index in brackets)
            l = int(re.findall('\[(.+?)\]', s)[0].replace('[', '').replace(']',''))
            
            p = (pos, pos + l)
            
            ls.append(l)
            
            p = np.digitize(list(p), [0] + list(pos_)) - 1
    
            pos += l
            positions.append(p)

            edges = []
            ix = n + 1
            order = []
            
            for node in list(t.levelorder())[::-1]:
                if node.is_tip():
                    node.age = 0.
                else:
                    c = node.children[0]
                    node.age = c.age + c.length
            
            for node in t.levelorder():
                if node.name is None:
                    node.name = ix
                    ix += 1
                    
                node.name = int(node.name) - 1
                order.append(node.name)
                
            A = np.zeros((2*n - 1, 2*n - 1))
            for node in t.levelorder():
                edges.extend([(node.name, u.name, u.length) for u in node.children])
            
            ages.append([u.age for u in t.levelorder()])
            orders.append(order)
            e.append(edges)
                
        ret['order'].append(orders)
        ret['edges'].append(e)
        ret['positions'].append(positions)
        ret['ages'].append(ages)
        
    return ret

def split(word):
    return [char for char in word]

def get_nx_distance_matrices(G):
    nodes = list(range(0, len(G.nodes())))
    indices = list(itertools.combinations(nodes, 2))
    
    paths = nx.shortest_path(G)
    D = np.array([len(paths[i][j]) for (i,j) in indices]) / 2.
    
    D_mut = []
    for i,j in indices:
        path = paths[i][j]
        
        _ = [G.edges[path[k], path[k + 1]]['n_mutations'] for k in range(len(path) - 1)]

        D_mut.append(sum(_))
            
    D_branch = []
    for i,j in indices:
        path = paths[i][j]
        
        _ = [G.edges[path[k], path[k + 1]]['weight'] for k in range(len(path) - 1)]

        D_branch.append(sum(_))

    D_r = []
    for i,j in indices:
        path = paths[i][j]
        
        _ = [G.edges[path[k], path[k + 1]]['r'] for k in range(len(path) - 1)]

        D_r.append(np.mean(_))

    # hops, mutations, branch lengths, and mean region size along shortest paths
    D = np.array([D, D_mut, D_branch, D_r], dtype = np.float32)



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
    #ms_chunks[-1] += ['\n']

    if ancFile is not None:
        anc_lines = [u.decode('utf-8') for u in ancFile.readlines()]
    else:
        anc_lines = None
        
    X = []
    Y = []
    P = []
    intros = []
    params = []
    
    for chunk in ms_chunks:
        line = chunk[0]
        params_ = list(map(float, line.replace('//', '').replace('\n', '').split('\t')))
        
        if '*' in line:
            intros.append(True)
        else:
            intros.append(False)
        
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)
        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]], dtype = np.uint8)
        
        if x.shape[0] == 0:
            X.append(None)
            Y.append(None)
            P.append(None)
            params.append(None)
            
            continue
        
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
            
        if len(pos) == x.shape[1] - 1:
            pos = np.array(list(pos) + [1.])
            
        assert len(pos) == x.shape[1]
        
        if n is not None:
            x = x[:n,:]
            y = y[:n,:]
            
        X.append(x)
        Y.append(y)
        P.append(pos)
        params.append(params_)
        
    return X, Y, P, params