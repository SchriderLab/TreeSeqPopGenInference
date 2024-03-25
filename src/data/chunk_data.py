# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
def chunks(lst, n):
    ret = []
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        ret.append(lst[i:i + n])
        
    return ret

import random

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--n_per", default = "250")
    parser.add_argument("--n_per_class", default = "100000")

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
    
    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if os.path.isdir(os.path.join(args.idir, u))] + [args.idir]

    counter = 0
    for idir in idirs:
        logging.info('working on {}...'.format(idir))
        
        tag = idir.split('/')[-1]
        #idir = os.path.join(idir, 'ms')
        
        ifiles = glob.glob(os.path.join(idir, '*.msOut.gz'))
        random.shuffle(ifiles)
        
        ifiles = ifiles[:int(args.n_per_class)]
        
        ifile_chunks = chunks(ifiles, int(args.n_per))
        for ix in range(len(ifile_chunks)):
            ifiles = ifile_chunks[ix]
            
            odir = os.path.join(args.odir, '{0:05d}'.format(counter))
            os.system('mkdir -p {}'.format(odir))
            
            for ifile in ifiles:
                ifile_ = tag + '.' + ifile.split('/')[-1]
                
                os.system('cp {0} {1}'.format(ifile, os.path.join(odir, ifile_)))
            
            counter += 1

if __name__ == '__main__':
    main()

