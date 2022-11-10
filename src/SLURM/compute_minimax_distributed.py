# -*- coding: utf-8 -*-
import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import glob
import itertools

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

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
    
    cmd = 'sbatch -t 04:00:00 --mem=16G --wrap "python3 src/models/estimate_kl_minimax.py --ij {} --idir {} --ofile {}"'

    ifiles = [u for u in sorted(glob.glob(os.path.join(args.idir, '*'))) if not '.' in u]
    
    coms = list(itertools.combinations(range(len(ifiles)), 2))

    for i, j in coms:
        ij = '{},{}'.format(i, j)
        
        cmd_ = cmd.format(ij, args.idir, os.path.join(args.odir, '{}_{}.txt'.format(i, j)))
        
        print(cmd_)
        os.system(cmd_)

    # ${code_blocks}

if __name__ == '__main__':
    main()
