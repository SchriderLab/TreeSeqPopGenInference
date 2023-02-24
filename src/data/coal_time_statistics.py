# -*- coding: utf-8 -*-

import os
import argparse
import logging

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import cv2
import glob

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
    parser.add_argument("--n_bins", default = "1000")

    parser.add_argument("--grid", default = "n01")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    # ${odir_del_block}

    return args

from scipy.interpolate import interp1d
import pickle

def main():
    args = parse_args()
    
    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    
    means = []
    stds = []
    
    maxs = []
    mins = []
    
    entropies = []
    
    print('reading for maxima and minima...')
    for ifile in ifiles:
        x = np.load(ifile)
        
        D = x['D']
        ll = x['P']
        
        maxs.append(np.max(D))
        mins.append(np.min(D))
        
        means.append(np.mean(D))
        stds.append(np.std(D))

        entropies.append(np.mean(-1 * ll))

    print('mean entropy: {}'.format(np.mean(entropies)))

    if args.grid == "n01":
        N = np.linspace(100., 1000., 128)
        
        plt.scatter(means, stds, c = N)
        plt.savefig(args.ofile.replace('.pkl', '_means_std.png'), dpi = 100)
        plt.close()
        
        plt.plot(N, entropies)
        plt.savefig(args.ofile.replace('.pkl', '_entropy.png'), dpi = 100)
        plt.close()
        
    bins = np.linspace(np.min(mins), np.max(maxs), int(args.n_bins))
    h = np.zeros(len(bins) - 1)
    
    count = 0
        
    print('computing histogram...')
    for ifile in ifiles:
        print(ifile)
        x = np.load(ifile)
        
        D = x['D']
        
        h += np.histogram(D.flatten(), bins, density = True)[0]
        count += 1
        
    h /= count
    h = np.cumsum(h)
    h /= np.max(h)
    
    x = bins[:-1] + np.diff(bins) / 2.
    ii = np.where(h > 0.)
    
    x = x[ii]
    h = h[ii]
    
    x = np.concatenate([np.zeros(1), x])
    h = np.concatenate([np.zeros(1), h])
    
    f = interp1d(x, h)
    
    pickle.dump(f, open(args.ofile, 'wb'))
    print('done!')
    
    """
    Dmax = np.max(np.log(D), axis = -1)
    Dmin = np.min(np.log(D), axis = -1)

    maxs.append(np.max(Dmax))
    mins.append(np.min(Dmin))
    
    mean.append(np.mean(D))
    var.append(np.std(D))
        
    print(h)
    plt.bar(bins[:-1] + np.diff(bins) / 2., h)
    plt.savefig('log_hist.png', dpi = 100)
    plt.close()
        
    plt.scatter(mean, var, c = l, cmap = 'viridis')
    plt.savefig('mean_std_a001.png', dpi = 100)
    plt.close()

    print(np.max(maxs))
    print(np.mean(maxs))
    print(np.std(maxs))
    print(np.mean(mins))
    print(np.median(mins))
    print(np.percentile(mins, 90))
    print(np.std(mins))
    print(np.min(mins))
    """
    # ${code_blocks}

if __name__ == '__main__':
    main()

