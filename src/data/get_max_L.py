# -*- coding: utf-8 -*-
import os
import argparse
import logging

import gzip

from data_functions import load_data
from format_genomat import find_files, format_matrix
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    # ${args}

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
    
    ifiles = []
    
    classes = sorted(os.listdir(args.idir))
    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    chunk_size = int(args.chunk_size)
    
    for c in classes:
        idir = os.path.join(args.idir, c)
        ifiles.extend([(c, u) for u in find_files(idir)])
        
    lengths = []
    for tag, ifile in ifiles:
        X, y, P, params = load_data(ifile)
        
        ls = [u.shape[1] for u in X]
        
        lengths.append(np.max(ls))
    
    print(np.max(lengths))

    # ${code_blocks}

if __name__ == '__main__':
    main()
