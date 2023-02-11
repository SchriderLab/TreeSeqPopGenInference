# -*- coding: utf-8 -*-
import os
import argparse
import logging
import numpy as np
import itertools

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_grid_points", default = "3")
    parser.add_argument("--n_replicates", default = "4096")

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
    
    cmd = 'sbatch -t 02:00:00 --mem=8G --wrap "python3 src/data/simulate_msprime_grid.py --N {0} --alpha0 {1} --alpha1 {2} --m {3} --ofile {4} --n_replicates {5}"'
    
    mu = 1e-4
    N = np.linspace(500, 1000, int(args.n_grid_points))
    alpha0 = np.linspace(0.01, 0.0125, int(args.n_grid_points))
    alpha1 = np.linspace(0.02, 0.025, int(args.n_grid_points))
    m12 = np.linspace(0.05, 0.3, int(args.n_grid_points))
    
    todo = list(itertools.product(N, alpha0, alpha1, m12))
    
    for ix in range(len(todo)):
        n, a1, a2, m = todo[ix]

        ofile = os.path.join(args.odir, '{0:06d}.npz'.format(ix))
        
        cmd_ = cmd.format(n, a1, a2, m, ofile, args.n_replicates)
        
        print(cmd_)
        os.system(cmd_)
        
    # ${code_blocks}

if __name__ == '__main__':
    main()

