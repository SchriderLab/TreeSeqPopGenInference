# -*- coding: utf-8 -*-
# binary, n_replicates, out file
CMD_NEUTRAL = '{} 104 {} 200000 -Pt {} {} -Pre {} {} -en 0.00936159 0 0.110131 -en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175 | tee {} && gzip {}'
# binary, n_replicates, position, out file
CMD_HARD = '{} 104 {} 200000 -Pt {} {} -Pre {} {} -en 0.00936159 0 0.110131 -en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175 -ws 0 -Pa 229.167 4583.33 -Pu 0 0.0218182 -x {} | tee {} && gzip {}'
CMD_SOFT = '{} 104 {} 200000 -Pt {} {} -Pre {} {} -en 0.00936159 0 0.110131 -en 0.012239 0 0.110131 -en 0.0153697 0 0.0955665 -en 0.0187762 0 0.0955665 -en 0.022483 0 0.134382 -en 0.0265165 0 0.134382 -en 0.030905 0 0.221054 -en 0.03568 0 0.221054 -en 0.0408755 0 0.3969 -en 0.0465286 0 0.3969 -en 0.0526795 0 0.68908 -en 0.0593727 0 0.68908 -en 0.0666568 0 1.01323 -en 0.0745882 0 1.01323 -en 0.0832186 0 1.24956 -en 0.0926045 0 1.24956 -en 0.102815 0 1.37902 -en 0.113924 0 1.37902 -en 0.126015 0 1.34656 -en 0.139175 0 1.34656 -en 0.153495 0 1.18725 -en 0.169068 0 1.18725 -en 0.186004 0 1.00113 -en 0.204437 0 1.00113 -en 0.22449 0 0.846248 -en 0.246309 0 0.846248 -en 0.270055 0 0.73786 -en 0.295895 0 0.73786 -en 0.324016 0 0.671364 -en 0.354619 0 0.671364 -en 0.387919 0 0.635331 -en 0.424152 0 0.635331 -en 0.463623 0 0.620869 -en 0.506632 0 0.620869 -en 0.553436 0 0.627949 -en 0.604255 0 0.627949 -en 0.659555 0 0.660491 -en 0.719727 0 0.660491 -en 0.785173 0 0.732284 -en 0.856359 0 0.732284 -en 0.933891 0 0.86023 -en 1.01837 0 0.86023 -en 1.11023 0 1.04845 -en 1.21017 0 1.04845 -en 1.31894 0 1.25027 -en 1.43734 0 1.25027 -en 1.56617 0 1.37179 -en 1.70645 0 1.37179 -en 1.85919 0 1.36135 -en 2.02542 0 1.36135 -en 2.20633 0 1.2175 -en 2.40295 0 1.2175 -en 2.61655 0 1.2175 -en 2.84902 0 1.2175 -ws 0 -Pa 229.167 4583.33 -Pu 0 0.0218182 -Pf 0 0.2 -x {} | tee {} && gzip {}'

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
    parser.add_argument("--n_replicates", default = "50")
    parser.add_argument("--n_jobs", default = "1000")
    
    parser.add_argument("--mode", default = "theta")

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
    
    theta_range = np.array([220, 2200])
    rho_range = np.array([1008.33, 3025])
    
    n_jobs = int(args.n_jobs)
    binary = os.path.join(os.getcwd(), 'discoal/discoal')
    
    slurm_cmd = 'sbatch -t 08:00:00 --mem=32G -o {0} --wrap "{1}"'

    odir = os.path.join(args.odir, 'neutral')
    os.system('mkdir -p {}'.format(odir))
    
    for ix in range(n_jobs):
        factor = np.random.uniform(0.5, 1.5, 2) 
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))
        
        cmd = CMD_NEUTRAL.format(binary, int(args.n_replicates), theta_range[0] * factor[0], theta_range[1] * factor[0],
                                 rho_range[0] * factor[1], rho_range[1] * factor[1],
                                 ofile, ofile)
        
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)

    # locations drawn uniformly
    x = np.random.uniform(0.45, 0.55, n_jobs)        
    odir = os.path.join(args.odir, 'hard')
    os.system('mkdir -p {}'.format(odir))
    
    for ix in range(n_jobs):
        factor = np.random.uniform(0.5, 1.5, 2) 

        x_ = x[ix]
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))
        
        cmd = CMD_HARD.format(binary, int(args.n_replicates), theta_range[0] * factor[0], theta_range[1] * factor[0],
                                 rho_range[0] * factor[1], rho_range[1] * factor[1],
                              x_, ofile, ofile)
       
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)
        
    # locations drawn uniformly
    x = np.random.uniform(0.45, 0.55, n_jobs)      
    odir = os.path.join(args.odir, 'soft')
    os.system('mkdir -p {}'.format(odir))
    
    for ix in range(n_jobs):
        factor = np.random.uniform(0.5, 1.5, 2)
        x_ = x[ix]
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))

        cmd = CMD_SOFT.format(binary, int(args.n_replicates), theta_range[0] * factor[0], theta_range[1] * factor[0],
                                 rho_range[0] * factor[1], rho_range[1] * factor[1],
                                 x_, ofile, ofile)
       
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
        
        factor = np.random.uniform(0.5, 1.5, 2)
        
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))

        cmd = CMD_HARD.format(binary, int(args.n_replicates), theta_range[0] * factor[0], theta_range[1] * factor[0],
                                 rho_range[0] * factor[1], rho_range[1] * factor[1],
                                 x, ofile, ofile)

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
        
        factor = np.random.uniform(0.5, 1.5, 2)
        
        slurm_out = os.path.join(odir, '{0:06d}_slurm.out'.format(ix))
        ofile = os.path.join(odir, '{0:06d}.msOut'.format(ix))

        cmd = CMD_SOFT.format(binary, int(args.n_replicates), theta_range[0] * factor[0], theta_range[1] * factor[0],
                                 rho_range[0] * factor[1], rho_range[1] * factor[1],
                                 x, ofile, ofile)
        
        cmd = slurm_cmd.format(slurm_out, cmd)
        
        print(cmd)
        os.system(cmd)
        
        ix += 1
    
    

    

    # ${code_blocks}

if __name__ == '__main__':
    main()