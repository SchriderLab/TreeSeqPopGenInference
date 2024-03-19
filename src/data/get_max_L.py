# -*- coding: utf-8 -*-
import os
import argparse
import logging

import gzip

from data_functions import load_data
from format_genomat import find_files, format_matrix
import numpy as np
import glob

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

from tqdm import tqdm

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
    
    ifiles = []
    
    classes = sorted(os.listdir(args.idir))
    
    for c in classes:
        idir = os.path.join(args.idir, c)
        ifiles.extend([(c, u) for u in find_files(idir)])
    
    if len(ifiles) == 0:
        ifiles = glob.glob(os.path.join(args.idir, '*.msOut.gz'))
        ifiles = list(zip([None for u in ifiles], ifiles))
        
    lengths = []
    for tag, ifile in tqdm(ifiles):
        X, y, P, params = load_data(ifile)
        
        X = [u for u in X if u is not None]
        ls = [u.shape[1] for u in X]
        
        lengths.append(np.max(ls))
    
    logging.info('have max number of sites: {}'.format(max(lengths)))

if __name__ == '__main__':
    main()
