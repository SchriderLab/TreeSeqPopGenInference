# -*- coding: utf-8 -*-
import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import h5py

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")

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

import numpy as np

def main():
    args = parse_args()
    
    ifile = h5py.File(args.ifile, 'r')
    
    keys = list(ifile.keys())

    N_trees = []
    for key in keys:
        mask = ifile[key]['mask']
        
        n_trees = mask.sum(-1)
        N_trees.extend(n_trees)
        
    print(np.median(N_trees))
    print(np.std(N_trees))
        

    # ${code_blocks}

if __name__ == '__main__':
    main()


