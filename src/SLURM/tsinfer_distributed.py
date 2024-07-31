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

    parser.add_argument("--n_sample", default = "None")
    
    parser.add_argument("--debug", action = "store_true")
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
    
    
    if args.slurm:
        cmd = 'sbatch --mem=16G -t 2-00:00:00 -o {2} --wrap "python3 src/data/ts_infer.py --idir {0} --ofile {1}"'
    else:
        cmd = 'python3 src/data/ts_infer.py --idir {0} --ofile {1}'
        
    idirs = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if not '.' in u])

    for ix in range(len(idirs)):
        if args.slurm:        
            cmd_ = cmd.format(idirs[ix], os.path.join(args.odir, '{0:04d}.hdf5'.format(ix)), os.path.join(args.odir, '{0:04d}_slurm.out'.format(ix)))
        else:
            cmd_ = cmd.format(idirs[ix], os.path.join(args.odir, '{0:04d}.hdf5'.format(ix)))
        
        print(cmd_)
        if not args.debug:
            os.system(cmd_)
            
if __name__ == '__main__':
    main()
