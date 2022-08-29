# -*- coding: utf-8 -*-
import os
import argparse
import logging

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
        else:
            os.system('rm -rf {}'.format(os.path.join(args.odir, '*')))
    # ${odir_del_block}

    return args

import numpy as np

def main():
    args = parse_args()
    
    cmd = 'sbatch --mem=8G -t 04:00:00 -o {2} --wrap "python3 src/data/rewrite_npz_file.py --ifile {0} --odir {1} --idn {3}"'
    ifile = np.load(args.ifile)
    keys = list(ifile.keys())
    idns = list(set([u.split('_')[-1] for u in keys if '_' in u]))

    for idn in idns:
        log_file = os.path.join(args.odir, '{}_slurm.out'.format(idn))
        
        cmd_ = cmd.format(args.ifile, args.odir, log_file, idn)
        print(cmd_)
        os.system(cmd_)
        
    # ${code_blocks}

if __name__ == '__main__':
    main()

