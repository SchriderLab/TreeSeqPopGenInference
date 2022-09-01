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
    
    ofile = h5py.File(args.ofile, 'w')
    ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    ifiles = glob.glob(os.path.join(args.ms_dir, '*.msOut.gz'))
    logging.info('have {} files to parse...'.format(len(ifiles)))
    
    tags = [u.split('/')[-1].split('.')[0] for u in ifiles]
    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    
    s0, s1 = pop_sizes
    
    for ii in range(len(ifiles)):
        tag = tags[ii]
        ifile = ifiles[ii]
        
        anc_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if (u.split('.')[-1] == 'anc' and tag == u.split('_')[0])])
        if len(anc_files) == 0:
            logging.info('ERROR: have no matching .anc files for {}...'.format(ifile))
            continue
        else:
            logging.info('have {} anc files...'.format(len(anc_files)))
        
        
        # load the genotype matrices that correspond to the trees
        logging.info('reading data, {}...'.format(ifile))
        x, y, p, intros = load_data(ifile, None)
        
        print(len(intros), len(x), len(anc_files))
        if ('mig12' in ifile) or ('mig21' in ifile):
            filter_zeros = True
        else:
            filter_zeros = False
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
            if (ix + 1) % 10 == 0:
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
            #snp_widths.append(l)
            
            Xs = []
            Edges = []
            infos = []
            
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
                
                try:
                    if ij != len(lines) - 1:
                        next_line = lines[ij + 1]
                        next_line = next_line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                        next_line = next_line.split(' ')[:-1]
                    
                        end_snp = int(next_line[0])
                    else:
                        end_snp = x[ix].shape[1]
                except:
                    break
    
                start_snp = int(line[0])
                l_ = end_snp - start_snp
                
                sk_nodes = dict()
                
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
                        regions.append((int(line[j + 3]), int(line[j + 4])))
                        
                        edges.append((parents[-1], nodes[-1]))
                except:
                    break
                    
                nodes.append(-1)
                master_nodes = copy.copy(nodes)
                
                lengths.append(0.)
                
                print(edges)
                
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
                    
                    for ix in range(len(cs)):
                        root_ete = cs[ix]
                        
                        # find the edges
                        n = root_ete.name
                        ii = [u for u in range(len(edges)) if edges[u][0] == n]
                        e = [edges[u] for u in ii]
                        l = [lengths[u] for u in ii]
                        
                        for e_, l_ in zip(e, l):
                            _.append(root_ete.add_child(Tree(), name = e_[1], dist = l_))
                            
                    cs = copy.copy(_)
                                                
                T = root_ete.get_tree_root()
                            
                    
                level_order = [u.name for u in list(root.levelorder())]
                #sys.exit()
                
                G = nx.DiGraph()
                G.add_edges_from(edges)
                
                
                    
                
                # slim adjacency representation we have for TreeGANs
                # for a potential RNN or CNN route
                G_ = G.to_undirected()
                
                # let's do this to save time in the cases we don't want this anyway
                if args.topological_order:
                    s0, s1 = pop_sizes        
                            
                    i0 = current_day_nodes[:s0]
                    i1 = current_day_nodes[s0:s0 + s1]
                    
                    ii = [u for u in level_order if u in i0] + [u for u in level_order if u in i1] + [u for u in level_order if not ((u in i0) or (u in i1))]
                    
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
                
                if s1 > 0:
                    for node in current_day_nodes[:s0]:
                        data[node] = np.array([0., 1., 0., 0., 0., 0.])
                    
                    for node in current_day_nodes[s0:s0 + s1]:
                        data[node] = np.array([0., 0., 1., 0., 0., 0.])
                else:
                    for node in current_day_nodes:
                        data[node] = np.array([0., 1., 0.])
                
                nodes = copy.copy(current_day_nodes)
                
                
                while len(data.keys()) < len(current_day_nodes) * 2:
                    _ = []
                    for node in nodes:
                        for j in range(len(edges)):
                            if edges[j][-1] == node:
                                p = edges[j][0]
                                break
                    
                    
                        if p in data.keys():
                            if s1 > 0:
                                lv = (data[node][1:3].astype(np.uint8).astype(bool) | data[p][-2:].astype(bool)).astype(np.uint8)
                                data[p][-2:] = lv.astype(np.float32)
                        
                        else:
                            if s1 > 0:
                                lv = data[node][1:3]
                            
                                # cumulatively add branch lengths for a time parameter
                                # note that current day nodes have t = 0.
                                data[p] = np.array([lengths[j] + data[node][0], 0., 0., 1.] + list(lv))
                            else:
                                data[p] = np.array([lengths[j] + data[node][0], 0., 1.])
                        
                        _.append(p)
                        
                    nodes = copy.copy(_)
                    
                Gs = G.subgraph(list(sorted(data.keys())))
                
                T_nodes = list(T.iter_descendants())
                T_names = [u.name for u in T_nodes]
                
                print(list(sorted(data.keys())))
                
                if not (-1 in list(sorted(data.keys()))):
                    root_name = max(list(sorted(data.keys())))
                    T = [u for u in T_nodes if u.name == root_name][0]
                    
                    T_nodes = list(T.iter_descendants())
                    T_names = [u.name for u in T_nodes]


                to_prune = []
                for node in sorted(data.keys()):
                    if Gs.in_degree(node) == Gs.out_degree(node) == 1:
                        to_prune.append(node)
                        
                to_prune = to_prune + [u for u in master_nodes if (u not in sorted(data.keys())) and (u in T_names)]
                        
                print(to_prune)
                to_prune = [u for u in T_nodes if u.name in to_prune] + [u for u in master_nodes if (u not in sorted(data.keys())) and (u in T_names)]
                
                T.prune(to_prune, True)
                
                print(T)
                sys.exit()
                
                    
                X = []
                for node in sorted(data.keys()):
                    X.append(data[node])
                    
                X = np.array(X)
                print(X.shape)
                
                if args.topological_order:
                    # topologically order nodes
                    X = X[indices,:]
                    # change the edge indexes to topologically (levelorder) ordered
                    # as we ordered the node features
                    edges = [(indices_[u], indices_[v]) for u,v in edges]
                
        
                Xs.append(X)
                
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
                
                edges = [(v,u) for u,v in edges]
                
                # make edges bi-directional
                if args.bidirectional:
                    edges = edges + [(v,u) for u,v in edges]
                edges = np.array(edges).T
                
                Edges.append(edges)
                infos.append(info_vec)
                
            if len(Xs) > 0:
                if ix < N:
                    ofile.create_dataset('{1}/{0}/x'.format(ix, tag), data = np.array(Xs), compression = 'lzf')
                    ofile.create_dataset('{1}/{0}/edge_index'.format(ix, tag), data = np.array(Edges).astype(np.int32), compression = 'lzf')
                    ofile.create_dataset('{1}/{0}/info'.format(ix, tag), data = np.array(infos), compression = 'lzf')
                else:
                    ofile_val.create_dataset('{1}/{0}/x'.format(ix - N, tag), data = np.array(Xs), compression = 'lzf')
                    ofile_val.create_dataset('{1}/{0}/edge_index'.format(ix - N, tag), data = np.array(Edges).astype(np.int32), compression = 'lzf')
                    ofile_val.create_dataset('{1}/{0}/info'.format(ix - N, tag), data = np.array(infos), compression = 'lzf')
            
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
                
          
    ofile.close()
    ofile_val.close()
    
    # ${code_blocks}

if __name__ == '__main__':
    main()