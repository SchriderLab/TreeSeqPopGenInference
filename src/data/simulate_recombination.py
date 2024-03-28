# -*- coding: utf-8 -*-
import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def chunks(lst, n):
    _ = []
    
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])
        
    return _

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "recom.runms.sh", help = "file of ms commands for running recombination rate estimation simulations")
    parser.add_argument("--chunk_size", default = "1000", help = "relevant for SLURM. how many simulation replicates to give each job submission")
    parser.add_argument("--slurm", action = "store_true", help = "are we using SLURM i.e. sbatch?")
    parser.add_argument("--factor", default = "1", help = "factor to multiply theta and rho by")
    parser.add_argument("--n", default = "50", help = "number of individuals in the sample")

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
        cmd = 'sbatch -t 24:00:00 --mem=8g --wrap "msdir/ms {4} {2} -t tbs -r tbs {3} < {0} | tee {1} && gzip {1}"'
    else:
        cmd = 'msdir/ms {4} {2} -t tbs -r tbs {3} < {0} | tee {1} && gzip {1}'
        
    lines = open(args.ifile, 'r').readlines()
    c = chunks(lines, int(args.chunk_size))
    
    factor = float(args.factor)
    
    for ix, chunk in enumerate(c):
        tbs_file = os.path.join(args.odir, '{0:05d}.tbs'.format(ix))
        w = open(tbs_file, 'w')
        ofile = os.path.join(args.odir, '{0:05d}.msOut'.format(ix))

        for c_ in chunk:
            c_ = c_.split()
            t = float(c_[4]) * factor
            r = float(c_[6]) * factor
            
            w.write(" ".join([str(t), str(r)]) + "\n")
        w.close()
        
        cmd_ = cmd.format(tbs_file, ofile, len(chunk), '20001', args.n)
        
        print(cmd_)
        os.system(cmd_)
        

    # ${code_blocks}

if __name__ == '__main__':
    main()

