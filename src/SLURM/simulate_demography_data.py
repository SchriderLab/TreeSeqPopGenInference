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
    parser.add_argument("--n_jobs", default = "400")
    parser.add_argument("--n_replicates_per", default = "250")
    
    parser.add_argument("--pop_sizes", default = "20,14")
    parser.add_argument("--factor", default = "")

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
    
    cmd = 'sbatch -t 08:00:00 --mem=8G --wrap "python3 src/data/simulate_demography.py {0} {1} {2}"'
    n = int(args.n_replicates_per)
    
    for ix in range(int(args.n_jobs)):
        odir = os.path.join(args.odir, '{0:04d}'.format(ix))
        
        cmd_ = cmd.format(odir, 'demo', n)
        print(cmd_)
        os.system(cmd_)
        
    # ${code_blocks}

if __name__ == '__main__':
    main()

