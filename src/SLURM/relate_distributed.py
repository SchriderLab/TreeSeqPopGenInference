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
    
    parser.add_argument("--L", default = "1100000")
    parser.add_argument("--mu", default = "1.2e-8")
    parser.add_argument("--r", default = "1e-8")
    parser.add_argument("--n_samples", default = "34")
    
    parser.add_argument("--debug", action = "store_true")

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

def main():
    args = parse_args()
    
    cmd = 'sbatch --mem=8G -t 2-00:00:00 -o {2} --wrap "python3 src/data/relate.py --idir {0} --odir {1} --L {3} --mu {4} --r {5} --n_samples {6}"'
    idirs = []
    
    # traverse root directory, and list directories as dirs and files as files
    # find leaf directories
    for root, dirs, files in os.walk(args.idir):
        for dr in dirs:
            directory = root + "/" + dr
            if len([sub for sub in os.listdir(directory) \
                    if os.path.isdir(directory+"/"+sub)]) == 0:
                idirs.append(directory)
                
    for idir in idirs:
        odir = os.path.join(args.odir, idir.replace(args.idir, ''))
        os.system('mkdir -p {}'.format(odir))
        
        log_file = os.path.join(odir, 'slurm.out')
        
        cmd_ = cmd.format(idir, odir, log_file, args.L, args.mu, args.r, args.n_samples)
        print(cmd_)
        if not args.debug:
            os.system(cmd_)
        
    # ${code_blocks}

if __name__ == '__main__':
    main()


