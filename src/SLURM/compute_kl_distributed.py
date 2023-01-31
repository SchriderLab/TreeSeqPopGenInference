# -*- coding: utf-8 -*-
import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}
import glob

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--sample_sizes", default = "64,64")

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

    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    cmd = 'sbatch -t 2-00:00:00 --mem=16G --wrap "python3 src/models/compute_kl_matrix.py --idir {} --i {} --ofile {} --sample_sizes {}"'

    for ix in range(len(ifiles)):
        cmd_ = cmd.format(args.idir, ix, os.path.join(args.odir, '{0:04d}.npz'.format(ix)), args.sample_sizes)
        
        print(cmd_)
        os.system(cmd_)

if __name__ == '__main__':
    main()
