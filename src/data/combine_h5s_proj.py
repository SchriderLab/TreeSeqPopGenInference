# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import h5py
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
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")

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
    
    ifiles = glob.glob(os.path.join(args.idir, '*.hdf5'))
    
    ifiles_train = [u for u in ifiles if not '_val.' in u]
    ifiles_val = [u for u in ifiles if '_val.' in u]
    
    ofile = h5py.File(args.ofile, 'w')
    ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')

    classes = args.classes.split(',')
    
    count = dict()    
    # counts for each class
    for c in classes:
        count[c] = 0

    means = []
    means1 = []
    x2s = []
    Ls = []

    # first compute means
    for ifile in ifiles:
        ifile = h5py.File(ifile, 'r')
        
        keys = list(ifile.keys())
        for key in keys:
            skeys = list(ifile[key].keys())
            
            count[key] += len(skeys)
            
            for skey in skeys:
                x = np.array(ifile[key][skey]['x'])
                x1 = np.array(ifile[key][skey]['x1'])
                x2 = np.array(ifile[key][skey]['x2'])

                Ls.append(x.shape[0])
                
                means.append(np.mean(x, axis = 0))
                means1.append(np.mean(x1, axis = 0))
                x2s.append(x2)
                
    x_mean = np.mean(np.array(means), axis = 0).reshape(1, -1)
    x1_mean = np.mean(np.array(means1), axis = 0).reshape(1, -1)
    
    x2_mean = np.mean(np.array(x2s))
    x2_std = np.std(np.array(x2s))
    
    max_L = np.max(Ls)
    min_L = np.min(Ls)
    med_L = np.median(Ls)
    
    print('have min, max, med for seq length: {}, {}, {}'.format(min_L, max_L, med_L))
    print('have counts: {}'.format(count))
    
    print('shapes (x, x1, x2): {}, {}, {}'.format(x_mean.shape, x1_mean.shape, x2_mean.shape))
    
    # variances
    x_var = []
    x1_var = []
    
    for ifile in ifiles:
        ifile = h5py.File(ifile, 'r')
        
        keys = list(ifile.keys())
        for key in keys:
            skeys = list(ifile[key].keys())
            
            for skey in skeys:
                x = np.array(ifile[key][skey]['x'])
                x1 = np.array(ifile[key][skey]['x1'])
                
                x_var.append(np.mean((x - x_mean)**2, axis = 0))
                x1_var.append(np.mean((x1 - x1_mean)**2, axis = 0))
                
    x_std = np.sqrt(np.mean(np.array(x_var), axis = 0).reshape(1, -1))
    x1_std = np.sqrt(np.mean(np.array(x1_var), axis = 0).reshape(1, -1))

    
    

    # ${code_blocks}

if __name__ == '__main__':
    main()

