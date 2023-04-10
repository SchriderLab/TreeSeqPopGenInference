# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
from mpi4py import MPI
import glob

from data_functions import load_data
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--regression", action = "store_true")
    
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
    
    # configure MPI
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        seg_sites = []

    idirs = [u for u in sorted(glob.glob(os.path.join(args.idir, '*'))) if os.path.isdir(u)]
    if comm.rank == 0:
        logging.info('found {} directories to read...'.format(len(idirs)))
        

    for ix in range(comm.rank, len(idirs), comm.size):
        logging.info('{0}: on {1}...'.format(comm.rank, ix))
        try:
            msFile = glob.glob(os.path.join(idirs[ix], '*.msOut.gz'))[0]
            tag = msFile.split('/')[-1].split('.')[0]
            tag_ = msFile.split('/')[-2]
        except:
            logging.info('no data found for {}...'.format(idirs[ix]))
            comm.send([None, None], dest = 0)
            continue
        
        X, Y, pos, params = load_data(msFile)
        seg_sites = [u.shape[1] for u in X]
        print('max segsites: {}'.format(max(seg_sites)))
        
        for k in range(len(X)):
            np.savez(os.path.join(args.odir, '{0}_{2}_{1:05d}.npz'.format(tag, k, tag_)), x = X[k], pos = pos[k])
            
    # ${code_blocks}

if __name__ == '__main__':
    main()

