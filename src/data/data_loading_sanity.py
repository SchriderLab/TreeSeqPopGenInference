# -*- coding: utf-8 -*-
import os
import argparse
import logging

from data_functions import load_data
from format_genomat import find_files, format_matrix

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "/overflow/dschridelab/ddray/four_problems/dros/mt2_4/eval_sims_i2/")
    parser.add_argument("--n_sample", default = "34")
    
    parser.add_argument("--pop_sizes", default = "20,14")
    parser.add_argument("--chunk_size", default = "4")
    parser.add_argument("--out_shape", default = "1,34,512")

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
        
        if any([u is None for u in X]):
            print(ifile)
            
        if any([u.shape[0] != int(args.n_sample) for u in X]):
            print(ifile)
            
        for ix, x in enumerate(X):
            x, p = format_matrix(x, P[ix], pop_sizes, out_shape = tuple(map(int, args.out_shape.split(','))), mode = 'seriate_match')
        
            if x is None:
                print(ix, ifile)
    # ${code_blocks}

if __name__ == '__main__':
    main()

