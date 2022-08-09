# -*- coding: utf-8 -*-
import os
import argparse
import logging

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import h5py
import numpy as np
import random

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "demography_trees.hdf5")

    parser.add_argument("--odir", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()
    
    ifile = h5py.File(args.ifile, 'r')
    classes = list(ifile.keys())
    
    # t_coal, mean_time, std_time, median_time, mean_branch_length, median_branch_length, std_branch_length, skew_branch_length, max_branch_length, min_branch_length, ...
    
    categories = np.array([[0.,1.], [1., 1.], [0., 1.]])
    n_keys = 100
    
    data = dict()
    for c in classes:
        data[c] = dict()
        
        data[c]['t_coal'] = []
        data[c]['mean_branch_length'] = []
        data[c]['std_branch_length'] = []
        data[c]['median_branch_length'] = []
        data[c]['mean_time'] = []
        data[c]['std_time'] = []

        keys = list(ifile[c].keys())
        random.shuffle(keys)

        for key in keys[:n_keys]:
            skeys = [u for u in list(ifile[c][key].keys()) if ('x' not in u) and ('A' not in u)]
            
            for skey in skeys:
                iv = np.array(ifile[c][key][skey]['info'])
                X = np.array(ifile[c][key][skey]['x'])
                
                data[c]['t_coal'].append(iv[0])
                data[c]['mean_branch_length'].append(iv[4])
                data[c]['median_branch_length'].append(iv[5])
                data[c]['std_branch_length'].append(iv[6])
                data[c]['mean_time'].append(iv[1])
                data[c]['std_time'].append(iv[2])
                
    quants = ['t_coal', 'mean_branch_length', 'median_branch_length', 'std_branch_length']
    
    mean_times = []
    std_times = []
    for c in classes:
        mean_times = mean_times + data[c]['mean_time']
        std_times = std_times + data[c]['std_time']
    
    print(np.mean(mean_times), np.mean(std_times))
    
    fig, axes = plt.subplots(nrows = 3, ncols = 4, sharex = True)
    for ix in range(3):
        for ij in range(4):
            axes[ix, ij].hist(data[c][quants[ij]], bins = 35)
            
            
    plt.savefig(args.ofile, dpi = 100)
    
                # ${code_blocks}

if __name__ == '__main__':
    main()

