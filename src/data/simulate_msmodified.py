# -*- coding: utf-8 -*-
import os, sys

import numpy as np
import argparse
import logging

from data_functions import writeTbsFile
import copy
import subprocess

# this function creates an array for writing to text that has the ms parameters
# from a CSV file produced via bootstrapped DADI runs
# the values expected in the CSV our given below
def parameters_df(df, ix, thetaOverRho, migTime, migProb, n):
    # estimated from ??
    u = 5.0e-9
    L = 10000
    
    # Isolation, migration model
    # estimated with DADI (default)
    # nu1 and nu2 (before)
    # nu1_0 and nu2_0 (after split)
    # migration rates (Nref_m12, Nref_m21)
    ll, aic, Nref, nu1_0, nu2_0, nu1, nu2, T, Nref_m12, Nref_m21 = df[ix]
    
    nu1_0 /= Nref
    nu2_0 /= Nref
    nu1 /= Nref
    nu2 /= Nref
    
    T /= (4*Nref / 15.)
    
    alpha1 = np.log(nu1/nu1_0)/T
    alpha2 = np.log(nu2/nu2_0)/T
    
    theta = 4 * Nref * u * L
    rho = theta / thetaOverRho
    
    migTime = migTime * T
    
    p = np.tile(np.array([theta, rho, nu1, nu2, alpha1, alpha2, 0, 0, T, T, migTime, 1 - migProb, migTime]), (n, 1)).astype(object)
    
    return p, ll, Nref

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_samples", default = "2048", help = "number of alignments to simulate per job")
    parser.add_argument("--n_jobs", default = "1", help = "number of jobs.  If your on a system without SLURM, then this should be left to 1 (default)")
    
    parser.add_argument("--ifile", default = "params.txt", help = "CSV file of bootstrapped demographic estimates. only applicable for the drosophila case; --model dros")
    
    parser.add_argument("--model", default = "dros", help = "model you'd like to simulate. current options our 'archie' and 'dros'")
    parser.add_argument("--direction", default = "ab", help = "directionality of migration. only applicable for the drosophila case; --model dros")
    parser.add_argument("--slurm", action = "store_true")
    
    parser.add_argument("--window_size", default = "5000", help = "size in base pairs of the region to simulate")
    
    parser.add_argument("--mt_range", default = "None")
    parser.add_argument("--t_range", default = "None")
    
    parser.add_argument("--n_grid_points", default = "5")
    
    parser.add_argument("--trees", action = "store_true")

    parser.add_argument("--odir", default = "None", help = "directory for the ms output and logs to be written to")
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

import itertools

def main():
    args = parse_args()
    
    # the supported models for this script
    if not (args.model in ['dros', 'archie']):
        print('not a supported model for this script...')
        print("valid options for --model: 'archie', 'dros'")
        return
    
    if args.model == 'dros':
        df = np.loadtxt(args.ifile)
    
    slurm_cmd = 'sbatch -t 1-00:00:00 --mem=8G -o {0} --wrap "{1}"'
    n = int(args.n_samples)
    L = int(args.window_size)
    
    counter = 0
    
    migTime = np.linspace(1e-3, 0.75, 6)
    migProb = np.linspace(0., 0.99, 6)
    rho = np.linspace(0.1, 0.3, 6)
    T = np.linspace(0.4, 1.7, 6)
    
    params = list(itertools.product(migTime, migProb, rho, T))
    
    df = df[np.where(df[:,0] > -2000)[0],:]
    df_mean = np.mean(df, axis = 0).reshape(1, -1)
    
    # size of the populations
    SIZE_A = 64
    SIZE_B = 64
    
    P, ll, Nref = parameters_df(df_mean, 0, 1., 0., 0., n)
    for p in params:
        migTime, migProb, rho, T = p
        
        odir = os.path.join(args.odir, 'iter{0:06d}'.format(counter))
        counter += 1
        
        os.system('mkdir -p {}'.format(odir))
        migProb = 1 - migProb
        
        #t_range = tuple(list(map(float, args.t_range.split(','))))
        T_ = copy.copy(P[:,-4])
        
        # rescale alpha1 and alpha2
        P[:,4] *= T_
        P[:,5] *= T_
        
        P[:,-4] = T
        P[:,-5] = T
        #P[:,-4] /= (4*Nref / 15.)
        
        # rescale alpha1 and alpha2
        P[:,4] /= P[:,-4]
        P[:,5] /= P[:,-4]
        
        P[:,-1] = migTime * P[:,-4]
        P[:,-3] = migTime * P[:,-4]
        
        P[:, 1] = P[:,0] / rho
        P[:,-2] = migProb
    
        if ('ba' in args.direction) or ('ab' in args.direction):
            writeTbsFile(np.concatenate([P, np.random.randint(0, 2**14, size = (P.shape[0], 3))], axis = 1), os.path.join(odir, 'mig.tbs'))
        else:
            writeTbsFile(np.concatenate([P[:,:-3], np.random.randint(0, 2**14, size = (P.shape[0], 1))], axis = 1), os.path.join(odir, 'mig.tbs'))
            
        
        if args.direction == 'ab':
            cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 -seeds tbs tbs tbs < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msdir/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
        elif args.direction == 'ba':
            cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 1 tbs -ej tbs 3 2 -seeds tbs tbs tbs < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msdir/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
        elif args.direction == 'none':
            cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -seed tbs < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msdir/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
        
        # example command:
        # 34 1 -t 58.3288 -r 365.8836 10000 -T -L -I 2 20 14 -n 1 18.8855 -n 2 0.05542 -eg 0 1 6.5160 -eg 0 2 -7.5960 -ma x 0.0 0.0 x -ej 0.66698 2 1 -en 0.66698 1 1 -es 0.02080 2 0.343619 -ej 0.02080 3 1 -seeds 12674 8050 3617
        
        # print the command, do the command, gzip the outputs
        cmd = "echo '{0}' && {0} && gzip mig.msOut".format(cmd)
        print('simulating for parameters: {}'.format(P[0]))
        sys.stdout.flush()
        
        fout = os.path.join(odir, 'slurm.out')
        cmd = slurm_cmd.format(fout, cmd)
        
        print(cmd)
        os.system(cmd)
        
        
        
               
if __name__ == '__main__':
    main()

