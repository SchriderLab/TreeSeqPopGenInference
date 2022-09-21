# -*- coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "/pine/scr/d/d/ddray/seln_data.npz")
    

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
    
    ifile = np.load(args.ifile, allow_pickle = True, encoding = 'latin1')
    keys = list(ifile.keys())
    
    x_keys = sorted([u for u in keys if 'xtrain' in u and '_' in u], key = lambda x: int(x.split('_')[0]))
    pos_keys = sorted([u for u in keys if 'postrain' in u and '_' in u], key = lambda x: int(x.split('_')[0]))
    y_keys = sorted([u for u in keys if 'ytrain' in u and '_' in u], key = lambda x: int(x.split('_')[0]))
    
    classes = list(range(5))
    
    for ix in range(len(x_keys)):
        print('working on {}...'.format(x_keys[ix]))
        
        x = list(ifile[x_keys[ix]])
        pos = list(ifile[pos_keys[ix]])
        
        y = list(ifile[y_keys[ix]])
        
        counts = [y.count(classes[k]) for k in range(len(classes))]
        
        print(counts)

    
    # ${code_blocks}

if __name__ == '__main__':
    main()
