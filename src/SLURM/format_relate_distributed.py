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
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ms_dir", default = "None")

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
    
    cmd = 'sbatch --mem=16G -t 1-00:00:00 --wrap "python3 src/data/format_relate.py --idir {0} --ms_dir {1} --ofile {2}"'
    idirs = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if not '.' in u])
    
    
    
    for ix in range(len(idirs)):
        ms_dir = os.path.join(args.ms_dir, idirs[ix].split('/')[-1])
        
        if os.path.exists(ms_dir):
            cmd_ = cmd.format(idirs[ix], ms_dir, os.path.join(args.odir, '{0:04d}.hdf5'.format(ix)))
            print(cmd_)
            os.system(cmd_)
        
    # ${code_blocks}

if __name__ == '__main__':
    main()



