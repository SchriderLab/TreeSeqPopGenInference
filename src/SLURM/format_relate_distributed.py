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
    parser.add_argument("--pop_sizes", default = "208,0")
    
    parser.add_argument("--topological_order", action = "store_true")
    parser.add_argument("--n_sample", default = "None")
    
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
    
    cmd = 'sbatch --mem=16G -t 2-00:00:00 -o {3} --wrap "python3 src/data/format_relate_v2.py --idir {0} --ms_dir {1} --ofile {2} --pop_sizes {4} --n_sample {5} --topological_order"'
    idirs = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if not '.' in u])
    
    if not args.topological_order:
        cmd = cmd.replace(' --topological_order', '')
    
    for ix in range(len(idirs)):
        ms_dir = os.path.join(args.ms_dir, idirs[ix].split('/')[-1])
        
        if os.path.exists(ms_dir):
            cmd_ = cmd.format(idirs[ix], ms_dir, os.path.join(args.odir, '{0:04d}.hdf5'.format(ix)), os.path.join(args.odir, '{0:04d}_slurm.out'.format(ix)), args.pop_sizes, args.n_sample)
            
            print(cmd_)
            if not args.only_print:
                os.system(cmd_)
        
    # ${code_blocks}

if __name__ == '__main__':
    main()



