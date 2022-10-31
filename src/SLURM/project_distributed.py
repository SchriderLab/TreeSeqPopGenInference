# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def even_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    _ = []
    
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])

    return _

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--batch_size", default = "8")
    parser.add_argument("--ckpt", default = "None")    
    
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--only_print", action = "store_true")

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
    
    cmd = 'sbatch --mem=16G -t 2-00:00:00 -o {2} --wrap "python3 src/models/project_images.py --ifiles {0} --ofile {1} --ckpt {2}"'
    
    ifiles = glob.glob(os.path.join(args.idir, '*/*.png'))
    
    chunks = even_chunks(ifiles, int(args.batch_size))
    for ix in range(len(chunks)):
        ifiles_ = ','.join(chunks[ix])
        ofile = os.path.join(args.odir, '{0:05d}.npz'.format(ix))
        
        cmd_ = cmd.format(ifiles_, ofile, args.ckpt)
        
        if not args.only_print:
            os.system(cmd_)
            
        print(cmd_)

if __name__ == '__main__':
    main()
