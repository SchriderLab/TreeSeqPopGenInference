# -*- coding: utf-8 -*-

# binary, n_replicates, out file
CMD_NEUTRAL = '{} 104 {} 200000 -Pt 220 2200 -Pre 1008.33 3025 -en 0.00936159 0 0.110131 -en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175 | tee {} && gzip {}'
# binary, n_replicates, position, out file
CMD_HARD = '{} 104 {} 200000 -Pt 220 2200 -Pre 1008.33 3025 -en 0.00936159 0 0.110131 -en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175 -ws 0 -Pa 229.167 4583.33 -Pu 0 0.0218182 -x {} | tee {} && gzip {}'
CMD_SOFT = '{} 104 {} 200000 -Pt 220 2200 -Pre 1008.33 3025 -en 0.00936159 0 0.110131 -en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175 -ws 0 -Pa 229.167 4583.33 -Pu 0 0.0218182 -Pf 0 0.2 -x {} | tee {} && gzip {}'

CMD_NEUTRAL_ = '{} 104 {} 200000 -Pt 123.60181818181817 1236.0181818181818 -Pre 566.5083333333333 1699.525 | tee {} && gzip {}'
CMD_HARD_ = '{} 104 {} 200000 -Pt 123.60181818181817 1236.0181818181818 -Pre 566.5083333333333 1699.525 -ws 0 -Pa 128.75189393939394 2575.0378787878785 -Pu 0 0.038834380194466105 -x {} | tee {} && gzip {}'
CMD_SOFT_ = '{} 104 {} 200000 -Pt 123.60181818181817 1236.0181818181818 -Pre 566.5083333333333 1699.525 -ws 0 -Pa 128.75189393939394 2575.0378787878785 -Pu 0 0.038834380194466105 -Pf 0 0.2 -x {} | tee {} && gzip {}'

# notes
"""
discoal_multipop 208 40 200000 -Pt 220 2200 -Pre 1008.33 3025 -en 0.00936159 0 0.110131 
-en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382
 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 
 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956 
 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 
 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 
 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 
 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 
 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023 
 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 
 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 
 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175
"""

# for x in 0.045454545454545456 0.13636363636363635 0.22727272727272727 0.3181818181818182 0.4090909090909091 0.5 
# 0.5909090909090909 0.6818181818181818 0.7727272727272727 0.8636363636363636 0.9545454545454546;
# hard sweep
"""
discoal_multipop 208 40 200000 -Pt 220 2200 -Pre 1008.33 3025 -en 0.00936159 0 0.110131 
-en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382 
 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 
 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956
 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 -en 0.169068 0 1.18725
 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786
 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869
 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284
 -en 0.933891 0 0.86023 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027
 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175
 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175 -ws 0 -Pa 229.167 4583.33 -Pu 0 0.0218182 -x $x
"""

# soft sweep
"""
discoal_multipop 208 40 200000 -Pt 220 2200 -Pre 1008.33 3025 -en 0.00936159 0 0.110131 
-en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382
 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908
 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956
 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725
 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248
 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331
 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949
 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023
 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 -en 1.56617 0 1.37179
 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175
 -en 2.84902 0 1.2175 -ws 0 -Pa 229.167 4583.33 -Pu 0 0.0218182 -Pf 0 0.2 -x $x
"""

import os
import argparse
import logging

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_jobs", default = "10000")
    parser.add_argument("--n_replicates", default = "10")
    parser.add_argument("--near", action = "store_true")
    
    parser.add_argument("--no_demo", action = "store_true")

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

    n_jobs = int(args.n_jobs)
    binary = os.path.join(os.getcwd(), 'discoal/discoal')
    
    slurm_cmd = 'sbatch -t 08:00:00 --mem=32G -o {0} --wrap "{1}"'

    odir = os.path.join(args.odir, 'neutral')
    os.system('mkdir -p {}'.format(odir))
    
    for ix in range(n_jobs):
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))
        
        if not args.no_demo:
            cmd = CMD_NEUTRAL.format(binary, int(args.n_replicates), ofile, ofile)
        else:
            cmd = CMD_NEUTRAL_.format(binary, int(args.n_replicates), ofile, ofile)
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)

    # locations drawn uniformly
    x = np.random.uniform(0.45, 0.55, n_jobs)        
    odir = os.path.join(args.odir, 'hard')
    os.system('mkdir -p {}'.format(odir))
    
    for ix in range(n_jobs):
        x_ = x[ix]
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))
        if not args.no_demo:
            cmd = CMD_HARD.format(binary, int(args.n_replicates), x_, ofile, ofile)
        else:
            cmd = CMD_HARD_.format(binary, int(args.n_replicates), x_, ofile, ofile)
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)
        
    # locations drawn uniformly
    x = np.random.uniform(0.45, 0.55, n_jobs)      
    odir = os.path.join(args.odir, 'soft')
    os.system('mkdir -p {}'.format(odir))
    
    for ix in range(n_jobs):
        x_ = x[ix]
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))
        if not args.no_demo:
            cmd = CMD_SOFT.format(binary, int(args.n_replicates), x_, ofile, ofile)
        else:
            cmd = CMD_SOFT_.format(binary, int(args.n_replicates), x_, ofile, ofile)    
        
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)

    # locations drawn uniformly      
    odir = os.path.join(args.odir, 'hard-near')
    os.system('mkdir -p {}'.format(odir))
    
    ix = 0
    while ix < n_jobs:

        x = np.random.uniform(0.55, 0.97)   
        if x >= 0.45 and x <= 0.55:
            continue
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))
        if not args.no_demo:
            cmd = CMD_HARD.format(binary, int(args.n_replicates), x, ofile, ofile)
        else:
            cmd = CMD_HARD_.format(binary, int(args.n_replicates), x, ofile, ofile)
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)
        
        ix += 1
        
    # locations drawn uniformly   
    odir = os.path.join(args.odir, 'soft-near')
    os.system('mkdir -p {}'.format(odir))
    
    ix = 0
    while ix < n_jobs:
        x = np.random.uniform(0.03, 0.97)   
        if x >= 0.45 and x <= 0.55:
            continue
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))
        if not args.no_demo:
            cmd = CMD_SOFT.format(binary, int(args.n_replicates), x, ofile, ofile)
        else:
            cmd = CMD_SOFT_.format(binary, int(args.n_replicates), x, ofile, ofile)
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)
        
        ix += 1

    # ${code_blocks}


if __name__ == '__main__':
    main()



