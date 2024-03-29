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
    parser.add_argument("--slurm", action = "store_true")

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
    
    slurm_cmd = 'sbatch --mem=16G --time=08:00:00 -o {3} --gres=gpu:1 --qos=gpu_access --partition=volta-gpu --wrap "python3 src/models/project_images.py --idir {0} --odir {1} --ckpt {2}"'
    cmd = "python3 src/models/project_images.py --ifiles {0} --ofile {1} --ckpt {2}"
    
    idirs = sorted([u for u in glob.glob(os.path.join(args.idir, '*')) if not '.' in u])
    
    for ix in range(len(idirs)):
        idir = idirs[ix]
        odir = os.path.join(args.odir, '{0:05d}'.format(ix))
        
        if not args.slurm:
            cmd_ = cmd.format(idir, odir, args.ckpt)
        else:
            cmd_ = slurm_cmd.format(idir, odir, args.ckpt, os.path.join(args.odir, '{0:05}_slurm.out'.format(ix)))
        
        if not args.only_print:
            os.system(cmd_)
            
        print(cmd_)

if __name__ == '__main__':
    main()
