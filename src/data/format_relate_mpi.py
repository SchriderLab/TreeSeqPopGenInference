# -*- coding: utf-8 -*-
import os
import argparse
import logging

import networkx as nx

import sys
import copy
import h5py
import numpy as np
import glob

from data_functions import load_data
from scipy.stats import skew

from skbio.tree import TreeNode

import matplotlib.pyplot as plt
import random

from ete3 import Tree
import itertools
from scipy.spatial.distance import squareform
import time

from mpi4py import MPI

"""
notes:
training_results/seln_rnn_i2/ --n_steps 1000 --lr 0.00001 --L 128 --n_gcn_iter 16 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4
training_results/seln_rnn_i3/ --n_steps 1000 --lr 0.00001 --L 92 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4
training_results/seln_rnn_i4/ --n_steps 1000 --lr 0.00001 --L 128 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4"
training_results/seln_rnn_i5/ --n_steps 1000 --lr 0.0001 --L 128 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4"

"""

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ms_dir", default = "None")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--n_samples", default = "None")
    parser.add_argument("--val_prop", default = "0.1")
    parser.add_argument("--pop_sizes", default = "20,14")
    
    parser.add_argument("--n_sample", default = "104")
    
    parser.add_argument("--topological_order", action = "store_true")
    parser.add_argument("--bidirectional", action = "store_true")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    # configure MPI
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        ofile = h5py.File(args.ofile, 'w')
        ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    ifiles = glob.glob(os.path.join(args.ms_dir, '*.msOut.gz'))
    logging.info('have {} files to parse...'.format(len(ifiles)))
    
    tags = [u.split('/')[-1].split('.')[0] for u in ifiles]
    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    
    s0, s1 = pop_sizes
    
    for ii in range(len(ifiles)):
        comm.Barrier()
        
        tag = tags[ii]
        ifile = ifiles[ii]
        
        anc_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if (u.split('.')[-1] == 'anc' and tag == u.split('_')[0])])
        if len(anc_files) == 0:
            logging.info('ERROR: have no matching .anc files for {}...'.format(ifile))
            continue
        else:
            logging.info('have {} anc files...'.format(len(anc_files)))
        
        
        if comm.rank == 0:
            t0 = time.time()
            
            # load the genotype matrices that correspond to the trees
            logging.info('reading data, {}...'.format(ifile))
            x, y, p, intros = load_data(ifile, None)
            del y    
        
            if len(x) == 0:
                logging.info('ERROR: have no genotype matrices! for {}...'.format(ifile))
                continue
        
        if args.n_samples == "None":
            N = len(anc_files)
        else:
            N = int(args.n_samples)
            
        N_val = int(N * float(args.val_prop))
        N = N - N_val
        
        times = []
        
        logging.info('have {0} training and {1} validation replicates...'.format(N, N_val))
        logging.info('writing...')
        for ix in range(N + N_val):
            comm.Barrier()
            
            if comm.rank == 0:
                logging.info('on replicate {}...'.format(ix))
                
            if not os.path.exists(anc_files[ix].replace('.anc', '.mut')):
                logging.info('ERROR: {} has no matching .mut file!...'.format(anc_files[ix]))
                continue
            
            anc_file = open(anc_files[ix], 'r')
            
            if args.n_sample != "None":
                n_sample = int(args.n_sample)
                
                current_day_nodes = list(np.random.choice(range(sum(pop_sizes)), n_sample, replace = False))
            else:
                current_day_nodes = list(range(sum(pop_sizes)))
                
            lines = anc_file.readlines()[2:]
            l = x[ix].shape[1]
     
            comm.Barrier()
            if comm.rank == 0:
                logging.info('done reading...')
            
            if comm.rank != 0:
                for ij in range(comm.rank - 1, len(lines), comm.size - 1):
                    line = lines[ij]
                    
                    nodes = []
                    parents = []
                    lengths = []
                    n_mutations = []
                    regions = []
                    
                    edges = []
                    
                    # new tree
                    line = line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                    line = line.split(' ')[:-1]
                    
                    try:
                        if ij != len(lines) - 1:
                            next_line = lines[ij + 1]
                            next_line = next_line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                            next_line = next_line.split(' ')[:-1]
                        
                            end_snp = int(next_line[0])
                        else:
                            end_snp = copy.copy(l)
                    except:
                        continue
        
                    start_snp = int(line[0])
                    l_ = end_snp - start_snp
                    
                    sk_nodes = dict()
                    mut_dict = dict()
                    try:
                        for j in range(2, len(line), 5):
                            nodes.append((j - 1) // 5)
                            
                            p = int(line[j])
                            if p not in sk_nodes.keys():
                                sk_nodes[p] = TreeNode(name = p)
                                
                            length = float(line[j + 1])
                            
                            if (j - 1) // 5 not in sk_nodes.keys():
                                sk_nodes[(j - 1) // 5] = TreeNode(name = (j - 1) // 5, parent = sk_nodes[p], length = length)
                                sk_nodes[p].children.append(sk_nodes[(j - 1) // 5])
                            else:
                                sk_nodes[(j - 1) // 5].parent = sk_nodes[p]
                                sk_nodes[(j - 1) // 5].length = length
                                sk_nodes[p].children.append(sk_nodes[(j - 1) // 5])
                                
                            parents.append(p)
                            lengths.append(float(line[j + 1]))
                            n_mutations.append(float(line[j + 2]))
                            
                            mut_dict[nodes[-1]] = n_mutations[-1]
                            regions.append((int(line[j + 3]), int(line[j + 4])))
                            
                            edges.append((parents[-1], nodes[-1]))
                    except:
                        break
                        
                    nodes.append(-1)
                    master_nodes = copy.copy(nodes)
                    
                    lengths.append(0.)
                    
                    root = None
                    for node in sk_nodes.keys():
                        node = sk_nodes[node]
                        if node.is_root():
                            root = node
                            break
                        
                    root_ete = Tree(name = -1)
                    
                    cs = []
                    for c in root.children:
                        cs.append(root_ete.add_child(Tree(), name = c.name, dist = c.length))
                    
                    while len(cs) > 0:
                        _ = []
                        
                        for j in range(len(cs)):
                            root_ete = cs[j]
                            
                            # find the edges
                            n = root_ete.name
                            ii = [u for u in range(len(edges)) if edges[u][0] == n]
                            e = [edges[u] for u in ii]
                            ls = [lengths[u] for u in ii]
                            
                            for e_, l_ in zip(e, ls):
                                _.append(root_ete.add_child(Tree(), name = e_[1], dist = l_))
                                
                        cs = copy.copy(_)
                                                    
                    data = dict()
                    if s1 > 0:
                        for node in current_day_nodes[:s0]:
                            data[node] = np.array([0., 1., 0., 0., 0., 0.])
                        
                        for node in current_day_nodes[s0:s0 + s1]:
                            data[node] = np.array([0., 0., 1., 0., 0., 0.])
                    else:
                        for node in current_day_nodes:
                            data[node] = np.array([0., 1., 0., mut_dict[node]])
                    
                    nodes = copy.copy(current_day_nodes)
                    
                    Tr = root_ete.get_tree_root()
                    master_nodes = list(Tr.iter_descendants()) + [Tr]
                    _ = [u.name for u in master_nodes]
                    
                    T = Tr.get_common_ancestor([u for u in Tr.get_descendants() if u.name in nodes])
                    
                    T_nodes = list(T.iter_descendants())
                    T_names = [u.name for u in T_nodes]
                    
                    T_present = [u for u in T_nodes if u.name in nodes]
                
                    to_prune = copy.copy(current_day_nodes)
                    
                    to_prune = [u for u in T_nodes if u.name in to_prune]
                    T.prune(to_prune, True)
    
                    T_nodes = list(T.iter_descendants()) + [T]
                    T_names = [u.name for u in T_nodes]
    
                    edges = []
                    while len(T_present) > 0:
                        _ = []
                        
                        for c in T_present:
                            c_ = c.name
                            branch_l = c.dist
                            
                            p = c.up
                            if p is not None:
                            
                                p = c.up.name
                                
                                if p not in data.keys():
                                    if p in mut_dict.keys():
                                        data[p] = np.array([data[c_][0] + branch_l, 0., 1., mut_dict[p]])
                                    else:
                                        data[p] = np.array([data[c_][0] + branch_l, 0., 1., 0.])
                                
                                    _.append(c.up)
                                
                                if p in T_names:
                                    edges.append((T_names.index(c_), T_names.index(p)))
                            
                            
                        T_present = copy.copy(_)
                        
                    ## Generating Distance matrices (via networkx)
                    ################ --
                    # ******************************************* #
                    # =========================================== #
                    lengths = dict(zip(range(len(T_names)), [lengths[u] for u in range(len(master_nodes)) if master_nodes[u].name in T_names]))
                    n_mutations = dict(zip(range(len(T_names)), [n_mutations[u] for u in range(len(master_nodes)) if master_nodes[u].name in T_names]))
                    regions = dict(zip(range(len(T_names)), [regions[u] for u in range(len(master_nodes)) if master_nodes[u].name in T_names]))
                    
                    Gu = nx.Graph()
                    for k in range(len(edges)):
                        child = edges[k][0]
                        
                        Gu.add_edge(edges[k][0], edges[k][1], weight = lengths[child], 
                                   n_mutations = n_mutations[child], hop = 0.5, r = regions[child][1] - regions[child][0],
                                   rp = np.mean(regions[k]))
        
                    for k in range(len(edges)):
                        child = edges[k][0]
                        
                        Gu.add_edge(edges[k][1], edges[k][0], weight = lengths[child], 
                                   n_mutations = n_mutations[child], hop = 0.5, r = regions[child][1] - regions[child][0],
                                   rp = np.mean(regions[k]))
                        
                    paths = nx.shortest_path(Gu)
    
                    indices = list(itertools.combinations(range(len(T_names)), 2))
                    D = np.array([len(paths[i][j]) for (i,j) in indices]) / 2.
                    D_mut = []
                    for i,j in indices:
                        path = paths[i][j]
                        
                        _ = [Gu.edges[path[k], path[k + 1]]['n_mutations'] for k in range(len(path) - 1)]
    
                        D_mut.append(sum(_))
                            
                    D_branch = []
                    for i,j in indices:
                        path = paths[i][j]
                        
                        _ = [Gu.edges[path[k], path[k + 1]]['weight'] for k in range(len(path) - 1)]
        
                        D_branch.append(sum(_))
        
                    D_r = []
                    for i,j in indices:
                        path = paths[i][j]
                        
                        _ = [Gu.edges[path[k], path[k + 1]]['r'] for k in range(len(path) - 1)]
        
                        D_r.append(np.mean(_))
                        
                    D = squareform(np.array(D))
                    D_mut = squareform(np.array(D_mut))
                    D_r = squareform(np.array(D_r))
                    D_branch = squareform(np.array(D_branch))
                        
                    X = []
                    for node in T_names:
                        X.append(data[node])
                        
                        
                    X = np.array(X)
                    
                    #print(len(edges), X.shape)
                    edges = edges[:X.shape[0]]
                    
                    if args.topological_order:
                        G = nx.DiGraph()
                        G.add_edges_from(edges)
                        
                        level_order = []
                        for node in T.traverse("levelorder"):
                            level_order.append(node.name)
                        # slim adjacency representation we have for TreeGANs
                        # for a potential RNN or CNN route
                        G_ = G.to_undirected()
                        
                        s0, s1 = pop_sizes        
                                
                        i0 = current_day_nodes[:s0]
                        i1 = current_day_nodes[s0:s0 + s1]
                        
                        ii = [u for u in level_order if u in i0] + [u for u in level_order if u in i1] + [u for u in level_order if not ((u in i0) or (u in i1))]
                        
                        A = np.array(nx.adjacency_matrix(G_, nodelist = ii).todense())
                        
                        i, j = np.tril_indices(A.shape[0])
                        A[i, j] = 0.
        
                        A = A[:,len(current_day_nodes):]
                        indices = [T_names.index(u) for u in ii]
                        indices_ = dict(zip(range(len(T_names)), [ii.index(u) for u in T_names]))
                        
                        lengths_ = np.array([lengths[u] for u in indices]).reshape(-1, 1)
                        
                        A = np.concatenate([A.astype(np.float32), lengths_], 1)
                        
                        # topologically order nodes
                        X = X[indices,:]
                        
                        # change the edge indexes to topologically (levelorder) ordered
                        # as we ordered the node features
                        edges = [(indices_[u], indices_[v]) for u,v in edges]
                    
                        D = D[np.ix_(indices, indices)]
                        D_mut = D_mut[np.ix_(indices, indices)]
                        D_branch = D_branch[np.ix_(indices, indices)]
                        D_r = D_r[np.ix_(indices, indices)]
                        
                    else:
                        A = None
                    
                    # hops, mutations, branch lengths, and mean region size along shortest paths
                    D = np.array([D, D_mut, D_branch, D_r], dtype = np.float32)

                    
                    ii = list(np.where(X[:,0] > 0)[0])
                    times.extend(X[ii,0])
                    
                    t_coal = np.max(X[:,0])
                    mean_time = np.mean(X[ii,0])
                    std_time = np.std(X[ii,0])
                    median_time = np.median(X[ii,0])
                    
                    lengths = list(lengths.values())
                    
                    mean_branch_length = np.mean(lengths)
                    median_branch_length = np.median(lengths)
                    std_branch_length = np.std(lengths)
                    skew_branch_length = skew(lengths)
                    max_branch_length = np.max(lengths)
                    position = ((start_snp + end_snp) / 2.) / l
                    w = l_ / l
                    
                    info_vec = np.array([t_coal, mean_time, std_time, median_time, mean_branch_length, median_branch_length, std_branch_length, skew_branch_length, max_branch_length,
                                         position, w, l])
                    
                    # make edges bi-directional
                    if args.bidirectional:
                        edges = edges + [(v,u) for u,v in edges]
                    edges = np.array(edges).T
                    
                    # X, edges, info_vec, D, A
                    comm.send([ij, X, edges, info_vec, D, A], dest = 0)
                    
                comm.send([None, None, None, None, None, None], dest = 0)
                    
            else:
                t0 = time.time()
                n_done = 0
                
                Xs = []
                Edges = []
                infos = []
                
                As = []
                Ds = []
                indices = []
                
                while n_done < comm.size - 1:
                    ij, X, edges, info_vec, D, A = comm.recv(source = MPI.ANY_SOURCE)
                    
                    if X is None:
                        n_done += 1
                        continue
                    
                    Xs.append(X)
                    Edges.append(edges)
                    infos.append(info_vec)
                    Ds.append(D)
                    indices.append(ij)
                    
                    if A is not None:
                        As.append(A)
            
                
            
                infos = np.array(infos)
                global_vec = np.array(list(np.mean(infos, axis = 0)) + list(np.std(infos, axis = 0)) + list(np.median(infos, axis = 0)) + [infos.shape[0]], dtype = np.float32)
                
                ii = np.argsort(indices)
                Xs = [Xs[u] for u in ii]
                Edges = [Edges[u] for u in ii]
                infos = [infos[u] for u in ii]
                if len(As) > 0:
                    As = [As[u] for u in ii]
                    
                Ds = [Ds[u] for u in ii]
                
                if len(Xs) > 0:
                    if ix < N:
                        ofile.create_dataset('{1}/{0}/global_vec'.format(ix, tag), data = global_vec, compression = 'lzf')
                        ofile.create_dataset('{1}/{0}/x'.format(ix, tag), data = np.array(Xs), compression = 'lzf')
                        ofile.create_dataset('{1}/{0}/edge_index'.format(ix, tag), data = np.array(Edges).astype(np.int32), compression = 'lzf')
                        ofile.create_dataset('{1}/{0}/info'.format(ix, tag), data = np.array(infos), compression = 'lzf')
                        ofile.create_dataset('{1}/{0}/D'.format(ix, tag), data = np.array(Ds), compression = 'lzf')
                    else:
                        ofile_val.create_dataset('{1}/{0}/global_vec'.format(ix - N, tag), data = global_vec, compression = 'lzf')
                        ofile_val.create_dataset('{1}/{0}/x'.format(ix - N, tag), data = np.array(Xs), compression = 'lzf')
                        ofile_val.create_dataset('{1}/{0}/edge_index'.format(ix - N, tag), data = np.array(Edges).astype(np.int32), compression = 'lzf')
                        ofile_val.create_dataset('{1}/{0}/info'.format(ix - N, tag), data = np.array(infos), compression = 'lzf')
                        ofile_val.create_dataset('{1}/{0}/D'.format(ix - N, tag), data = np.array(Ds), compression = 'lzf')
                
                Xg = x[int(anc_files[ix].split('/')[-1].split('.')[0].split('chr')[-1]) - 1]
                A = np.array(As)
                
                if ix < N:
                    ofile.create_dataset('{1}/{0}/x_0'.format(ix, tag), data = Xg.astype(np.uint8), compression = 'lzf')
                    if A.shape[0] > 0:
                        ofile.create_dataset('{1}/{0}/A'.format(ix, tag), data = A, compression = 'lzf')
                    ofile.flush()
                else:
                    ofile_val.create_dataset('{1}/{0}/x_0'.format(ix - N, tag), data = Xg.astype(np.uint8), compression = 'lzf')
                    if A.shape[0] > 0:
                        ofile_val.create_dataset('{1}/{0}/A'.format(ix - N, tag), data = A, compression = 'lzf')
                    ofile_val.flush()
                    
                logging.info('iteration with {} graphs took {} seconds...'.format(len(Xs), time.time() - t0))
                
    if comm.rank == 0:      
        ofile.close()
        ofile_val.close()
    
    # ${code_blocks}

if __name__ == '__main__':
    main()