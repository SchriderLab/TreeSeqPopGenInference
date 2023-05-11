# -*- coding: utf-8 -*-
import os
import argparse
import logging
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
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--n_samples", default = "2500")

    parser.add_argument("--regression", action = "store_true")

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
    v2s = []
    ys = []
    
    n_samples = int(args.n_samples)
    
    random.shuffle(keys)
    
    _ = []
    ls = []
    for key in keys[:n_samples]:
       
        if 'x' in list(ifile[key].keys()):
            x = np.array(ifile[key]['x'])
            v = np.array(ifile[key]['x1'])
            v2 = np.array(ifile[key]['global_vec'])
            mask = np.array(ifile[key]['mask'])

            if args.regression:
                ys.append(np.log(np.array(ifile[key]['y'])[0,0].flatten()))

            ls.append(x.shape[0])
            
            x_ = x[:,:,:,0]
            x_ = np.log(x_[np.where(x_ > 0.)])
            for k in range(len(v)):
                m = mask[k]
                v_ = v[k,np.where(m == 1)]
                
                vs.append(v_[0])
            
            v2s.extend(list(v2))
                
            _.extend(x_)
            
    if args.regression:
        ys = np.array(ys)
        y_mean = np.mean(ys, axis = 0)
        y_std = np.std(ys, axis = 0)
            
    vs = np.concatenate(vs)
    v2s = np.array(v2s)
        
    v_mean = np.mean(vs, axis = 0)
    v_std = np.std(vs, axis = 0)
    
    v2_mean = np.mean(v2s, axis = 0)
    v2_std = np.std(v2s, axis = 0)
    
    if args.regression:
        np.savez(args.ofile, v_mean = v_mean, v_std = v_std, v2_mean = v2_mean, v2_std = v2_std, 
                     times = np.array([np.mean(_), np.std(_)]), y_mean = y_mean, y_std = y_std)
    else:
        np.savez(args.ofile, v_mean = v_mean, v_std = v_std, v2_mean = v2_mean, v2_std = v2_std, 
                     times = np.array([np.mean(_), np.std(_)]))
    
    print(v_mean, v_std)
    print(v2_mean, v2_std)
    print(np.mean(_), np.std(_), np.max(_), np.min(_), np.mean(ls), np.min(ls), np.max(ls))
    print(y_mean, y_std)
    # ${code_blocks}

if __name__ == '__main__':
    main()

