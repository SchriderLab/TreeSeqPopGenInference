# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def chunks(lst, n):
    _ = []
    
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])
        
    return _

def rejection_sample(scale = 661038.9851980766 ** (-1)):
    while True:
        x = np.random.exponential(scale)
        if x > 10**(-8) and x < 10**-6:
            return x

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_replicates", default = "1000", help = "relevant for SLURM. how many simulation replicates to give each job submission or each call to ms")
    parser.add_argument("--n_jobs", default = "100", help = "number of ms commands to run / jobs to run")
    
    parser.add_argument("--slurm", action = "store_true", help = "are we using SLURM i.e. sbatch?")
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
    
    N = [5e3, 10e3, 15e3, 20e3, 50e3]
    mu = 1.5e-8
    
    for ix in range(int(args.n_jobs)):
        tbs_file = os.path.join(args.odir, '{0:05d}.tbs'.format(ix))
        w = open(tbs_file, 'w')
        
        ofile = os.path.join(args.odir, '{0:05d}.msOut'.format(ix))

        for ix in range(int(args.n_replicates)):
            N_ = np.random.choice(N)
            
            t = 4 * N_ * mu * 20000 
            r = 4 * N_ * rejection_sample() * 20000
            
            w.write(" ".join([str(t), str(r)]) + "\n")
        w.close()
        
        cmd_ = cmd.format(tbs_file, ofile, int(args.n_replicates), '20001', args.n)
        
        print(cmd_)
        os.system(cmd_)
        
if __name__ == '__main__':
    main()
    
    