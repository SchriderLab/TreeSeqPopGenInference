# -*- coding: utf-8 -*-
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
from mpi4py import MPI

import matplotlib.pyplot as plt
import random

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
    parser.add_argument("--val_prop", default = "0.05")
    parser.add_argument("--pop_sizes", default = "20,14")

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
        ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    comm.Barrier()
    if comm.rank != 0:
        ifiles = glob.glob(os.path.join(args.ms_dir, '*/*.msOut.gz'))
        
        tags = [u.split('/')[-1].split('.')[0] for u in ifiles]
        pop_sizes = list(map(int, args.pop_sizes.split(',')))
        
        comm.Barrier()
        
        for ii in range(len(ifiles)):
            tag = tags[ii]
            ifile = ifiles[ii]
            
            idir = os.path.join(args.idir, ifile.split('/')[-2])
            
            anc_files = sorted([os.path.join(idir, u) for u in os.listdir(idir) if (u.split('.')[-1] == 'anc' and tag in u)])
            
            # for now the mutations are un-used...
            # this would have to be formatted as an edge feature would could be done really easily
            # I think we don't read these cause theyre reduntant actually?
            mut_files = sorted([os.path.join(idir, u) for u in os.listdir(idir) if (u.split('.')[-1] == 'mut' and tag in u)])
            
            indices_f = list(range(len(anc_files)))
            random.shuffle(indices_f)
            
            anc_files = [anc_files[u] for u in indices_f]
            mut_files = [mut_files[u] for u in indices_f]
            
            # load the genotype matrices that correspond to the trees
            logging.info('reading data, {}...'.format(ifile))
            x, y, p = load_data(ifile, None)
            del y
            
            if args.n_samples == "None":
                N = len(x)
            else:
                N = int(args.n_samples)
                
            N_val = int(N * float(args.val_prop))
            N = N - N_val
            
            branch_lengths = []
            snp_widths = []
            
            times = []
            
            comm.Barrier()
            
            logging.info('{2}: have {0} training examples to send and {1} validation samples...'.format(N, N_val, comm.rank))
            for ix in range(comm.rank - 1, N + N_val, comm.size - 1):
                #if (ix + 1) % 100 == 0:
                #    print(ix)
                
                if ix > N:
                    val = True
                else:
                    val = False
                
                anc_file = open(anc_files[ix], 'r')
                
                lines = anc_file.readlines()[2:]
                
                l = x[ix].shape[1]
                snp_widths.append(l)
                
                As = []
                for ij in range(len(lines)):
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
                    
                    if ij != len(lines) - 1:
                        next_line = lines[ij + 1]
                        next_line = next_line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                        next_line = next_line.split(' ')[:-1]
                    
                        end_snp = int(next_line[0])
                    else:
                        end_snp = x[ix].shape[1]
        
                    start_snp = int(line[0])
                    l_ = end_snp - start_snp
                    
                    sk_nodes = dict()
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
                        regions.append((int(line[j + 3]), int(line[j + 4])))
                        
                        edges.append((parents[-1], nodes[-1]))
                        
                    nodes.append(-1)
                    lengths.append(0.)
                    
                    root = None
                    for node in sk_nodes.keys():
                        node = sk_nodes[node]
                        if node.is_root():
                            root = node
                            break
                        
                    level_order = [u.name for u in list(root.levelorder())]
                    #sys.exit()
                    
                    G = nx.DiGraph()
                    G.add_edges_from(edges)
                    
                    current_day_nodes = []
                    # find the nodes which have no out degree
                    for node in G.nodes():
                        d = G.out_degree(node)
                        
                        if d == 0:
                            current_day_nodes.append(node)
                            
                    s0, s1 = pop_sizes        
                            
                    i0 = current_day_nodes[:s0]
                    i1 = current_day_nodes[s0:s0 + s1]
                    
                    ii = [u for u in level_order if u in i0] + [u for u in level_order if u in i1] + [u for u in level_order if not ((u in i0) or (u in i1))]
                    
                    # slim adjacency representation we have for TreeGANs
                    # for a potential RNN or CNN route
                    G_ = G.to_undirected()
                    A = np.array(nx.adjacency_matrix(G_, nodelist = ii).todense())
                    
                    i, j = np.tril_indices(A.shape[0])
                    A[i, j] = 0.
    
                    A = A[:,len(current_day_nodes):]
                    indices = [nodes.index(u) for u in ii]
                    indices_ = dict(zip(nodes, [ii.index(u) for u in nodes]))
                    
                    lengths_ = np.array([lengths[u] for u in indices]).reshape(-1, 1)
                    
                    A = np.concatenate([A.astype(np.float32), lengths_], 1)
                    
                    As.append(A)
                                                    
                    data = dict()
                    for node in current_day_nodes[:s0]:
                        data[node] = np.array([0., 1., 0., 0., 0., 0.])
                    
                    for node in current_day_nodes[s0:s0 + s1]:
                        data[node] = np.array([0., 0., 1., 0., 0., 0.])
                        
                    t = 0.
                    
                    nodes = copy.copy(current_day_nodes)
                    while len(data.keys()) < len(G.nodes()):
                        _ = []
                        for node in nodes:
                            for j in range(len(edges)):
                                if edges[j][-1] == node:
                                    p = edges[j][0]
                                    break
                        
                            if p in data.keys():
                                lv = (data[node][1:3].astype(np.uint8).astype(bool) | data[p][-2:].astype(bool)).astype(np.uint8)
                                data[p][-2:] = lv.astype(np.float32)
                            else:
                                lv = data[node][1:3]
                            
                                # cumulatively add branch lengths for a time parameter
                                # note that current day nodes have t = 0.
                                data[p] = np.array([np.log(data[node][0] + lengths[j]), 0., 0., 1.] + list(lv))
                            
                            _.append(p)
                            
                        nodes = copy.copy(_)
                        
                    X = []
                    for node in sorted(data.keys()):
                        X.append(data[node])
                        
                    X = np.array(X)
                    # topologically order nodes
                    X = X[indices,:]
                    
                    lengths = np.array(lengths)
                    lengths = np.log(lengths[lengths > 0])
                    
                    # for stats to save later
                    branch_lengths.extend(np.random.choice(lengths, 10, replace = False))
                    
                    ii = list(np.where(X[:,0] > 0)[0])
                    
                    times.extend(X[ii,0])
                    
                    t_coal = np.max(X[:,0])
                    mean_time = np.mean(X[ii,0])
                    std_time = np.std(X[ii,0])
                    median_time = np.median(X[ii,0])
                    
                    mean_branch_length = np.mean(lengths)
                    median_branch_length = np.median(lengths)
                    std_branch_length = np.std(lengths)
                    skew_branch_length = skew(lengths)
                    max_branch_length = np.max(lengths)
                    min_branch_length = np.min(lengths)
                    position = ((start_snp + end_snp) / 2.) / l
                    w = l_ / l
                    
                    info_vec = np.array([t_coal, mean_time, std_time, median_time, mean_branch_length, median_branch_length, std_branch_length, skew_branch_length, max_branch_length, min_branch_length,
                                         position, w, l])
                    
                    # change the edge indexes to topologically (levelorder) ordered
                    # as we ordered the node features
                    edges = [(indices_[u], indices_[v]) for u,v in edges]
                    
                    # make edges bi-directional
                    edges = edges + [(v,u) for u,v in edges]
                    edges = np.array(edges).T
                    
                    comm.send([ix, ij, x, edges, info_vec, tag, val], dest = 0)
                
                Xg = x[indices_f[ix]]
                A = np.array(As)
                
                comm.send([ix, Xg, A, tag, val], dest = 0)
        
        comm.Barrier()
        
        comm.send(["done"], dest = 0)
    
    else:
        n_done = 0
        
        while n_done != comm.size - 1:
            _ = comm.recv(source = MPI.ANY_SOURCE)
            
            
            if len(_) == 7:
                ix, ij, x, edges, info_vec, tag, val = _
                
                if not val:
                    ofile.create_dataset('{2}/{0}/{1}/x'.format(ix, ij, tag), data = X, compression = 'lzf')
                    ofile.create_dataset('{2}/{0}/{1}/edge_index'.format(ix, ij, tag), data = edges.astype(np.int32), compression = 'lzf')
                    ofile.create_dataset('{2}/{0}/{1}/info'.format(ix, ij, tag), data = info_vec, compression = 'lzf')
                else:
                    ofile_val.create_dataset('{2}/{0}/{1}/x'.format(ix - N, ij, tag), data = X, compression = 'lzf')
                    ofile_val.create_dataset('{2}/{0}/{1}/edge_index'.format(ix - N, ij, tag), data = edges.astype(np.int32), compression = 'lzf')
                    ofile_val.create_dataset('{2}/{0}/{1}/info'.format(ix - N, ij, tag), data = info_vec, compression = 'lzf')
            elif len(_) == 5:
                ix, Xg, A, tag, val = _
                
                if ix < N:
                    ofile.create_dataset('{1}/{0}/x_0'.format(ix, tag), data = Xg.astype(np.uint8), compression = 'lzf')
                    ofile.create_dataset('{1}/{0}/A'.format(ix, tag), data = A, compression = 'lzf')
                    ofile.flush()
                else:
                    ofile_val.create_dataset('{1}/{0}/x_0'.format(ix - N, tag), data = Xg.astype(np.uint8), compression = 'lzf')
                    ofile_val.create_dataset('{1}/{0}/A'.format(ix - N, tag), data = A, compression = 'lzf')
                    ofile_val.flush()
            else:
                n_done += 1
                
            
                
        ofile.close()
        ofile_val.close()
    
    # ${code_blocks}

if __name__ == '__main__':
    main()


