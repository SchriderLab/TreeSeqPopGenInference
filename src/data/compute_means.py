# -*- coding: utf-8 -*-
import os
import argparse
import logging
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
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--ofile", default = "intro_means.npz")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
    
    vs = []
    _ = []
    ls = []
    for key in keys:
        print(key)
       
        if 'x' in list(ifile[key].keys()):
            x = np.array(ifile[key]['x'])
            v = np.array(ifile[key]['info'])
            
            ls.append(x.shape[0])
            
            x_ = x[:,:,:,0]
            x_ = np.log(x_[np.where(x_ > 0.)])
            
            vs.append(v)
            _.extend(x_)
            
    vs = np.concatenate(vs)
        
    v_mean = np.mean(vs, axis = 0)
    v_std = np.std(vs, axis = 0)
    
    np.savez(args.ofile, v_mean = v_mean, v_std = v_std, times = np.array([np.mean(_), np.std(_)]))
    
    print(v_mean, v_std)
    print(np.mean(_), np.std(_), np.max(_), np.min(_), np.mean(ls), np.min(ls), np.max(ls))
    # ${code_blocks}

if __name__ == '__main__':
    main()

