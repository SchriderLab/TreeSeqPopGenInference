# -*- coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np
import h5py

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    
    parser.add_argument("--classi", action = "store_true")

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
    
    ifile = h5py.File(args.ifile, 'r')
    
    count = 0
    
    keys = list(ifile.keys())
    if args.classi:
        for key in keys:
            keys_ = ifile[key].keys()
            
            for key_ in keys_:
                x = ifile[key][key_]['x']
           
                count += x.shape[0]
    
        logging.info('have {} replicates in file'.format(count))
        return
    
    if len(keys) == 1:
        keys = [keys[0] + '/' + u for u in list(ifile[keys[0]].keys())]

        count = len(keys)
        logging.info('have {} replicates in file'.format(count))
        
        return

    for key in keys:
        x = ifile[key]['x']
   
        count += x.shape[0]
        
    print(x.shape)
        
    logging.info('have {} replicates in file'.format(count))

    # ${code_blocks}

if __name__ == '__main__':
    main()


