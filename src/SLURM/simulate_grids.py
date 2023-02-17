# -*- coding: utf-8 -*-
import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import numpy as np
import itertools
import json
import sys

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--grid", default = "m01")
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
        else:
            os.system('rm -rf {}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()

    ifiles = []
    if args.grid == "m01":
        m01 = np.linspace(0.01, 0.5, 8)
        m10 = np.linspace(0.01, 0.5, 8)
        
        N = [1000, 1000]
        alpha = [0.01, 0.01]
        s = [32, 32]
    
        todo = list(itertools.product(m01, m10))
        
        for ix, (m1, m2) in enumerate(todo):
            ofile = os.path.join(args.odir, '{0:04d}.json'.format(ix))
            d = {"N" : N, "alpha" : alpha, "s" : s, "M" : [0., m1, 0., m2]}
        
            # Serializing json
            json_object = json.dumps(d, indent=4)
            
            # Writing to sample.json
            with open(ofile, "w") as outfile:
                outfile.write(json_object)
    
            ifiles.append(ofile)
    
    if len(ifiles) > 0:    
        cmd = 'sbatch -t 02:00:00 --mem=8G --wrap "src/data/simulate_msprime_grid.py --ifile {0} --ofile {1} --n_replicates {2}"'

        for ix in range(len(ifiles)):
            cmd_ = cmd.format(ifiles[ix], ifiles[ix].replace('.json', '.npz'), args.n_replicates)
        
            print(cmd_)
            os.system(cmd_)

    # ${code_blocks}

if __name__ == '__main__':
    main()
