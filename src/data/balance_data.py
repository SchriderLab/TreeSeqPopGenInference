# -*- coding: utf-8 -*-
import os
import argparse
import logging

from format_genomat import find_files
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
    
    classes = sorted(os.listdir(args.idir))
    
    ifiles = dict()
    for c in classes:
        idir = os.path.join(args.idir, c)
        ifiles[c] = find_files(idir)
        
    # min size
    l = min([len(ifiles[c]) for c in classes])
    
    for c in classes:
        ifiles_ = ifiles[c]
        
        l_ = len(ifiles_)
        n_del = l_ - l

        if n_del > 0:
            to_del = np.random.choice(ifiles_, n_del, replace = False)        
            
            cmd = 'rm {}'.format(' '.join(to_del))
            
            os.system(cmd)
    # ${code_blocks}

if __name__ == '__main__':
    main()

    
