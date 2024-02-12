# -*- coding: utf-8 -*-
import os
import argparse
import logging

from collections import OrderedDict
import glob

from data_functions import load_data
import random
import string

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

# selection params
# L = 110000
# mu = 1.2e-8
# r = 1e-8
# 

# for recombination experiments
# L = 20000
# mu = 1.5e-8
# r = 1e-7
# N = 14714

# dros params
# L = 10000
# mu = 5e-9
# r = 2e-8
# N = 233863

# demographic regression
# L = 1500000
# u = 1.2e-9
# r = 1e-8
# N = 1000

import re

def id_generator(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--L", default = "10000")
    parser.add_argument("--mu", default = "5e-8")
    parser.add_argument("--r", default = "2e-8")
    parser.add_argument("--N", default = "233863")
    
    parser.add_argument("--n_samples", default = "34")
    parser.add_argument("--relate_path", default = "relate/bin/Relate")

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
    
    idir = args.idir
    odir = args.odir
    
    logging.info('working on {}...'.format(idir))
    
    ### get haplotype and sample files
    logging.info('uncompressing data...')
    ifiles = [os.path.join(idir, u) for u in os.listdir(idir) if (('.ms' in u) or ('.msOut' in u))]
    print(ifiles)
    for ix in range(len(ifiles)):
        ifile = ifiles[ix]

        if '.gz' in ifile:
            os.system('gzip -d {}'.format(ifile))
            
            ifile = ifile.replace('.gz', '')
            
        ifiles[ix] = os.path.abspath(ifile)
    
    rscript_path = os.path.join(os.getcwd(), 'src/data/ms2haps.R')
    args.relate_path = os.path.join(os.getcwd(), args.relate_path)
    
    rcmd = 'cd {3} && Rscript ' + rscript_path + ' {0} {1} {2}'
    relate_cmd = 'cd {6} && ' + args.relate_path + ' --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'
    
    for ifile in ifiles:
        m_ofile = os.path.join(odir, ifile.split('/')[-1].replace('.msOut', '.anc'))
        
        tag = ifile.split('/')[-1].split('.')[0]
        
        logging.info('working on {}...'.format(ifile))
        logging.info('converting to haps / sample files via Rscript...')
        cmd_ = rcmd.format(ifile, tag, int(args.L), odir)
        os.system(cmd_)        
        
        L = float(args.L)  
        r = float(args.r)
        mu = float(args.mu)
        
        map_file = ifile.replace('.msOut', '.map')
        logging.info('writing map file {}...'.format(map_file))
        
        ofile = open(map_file, 'w')
        ofile.write('pos COMBINED_rate Genetic_Map\n')
        ofile.write('0 {} 0\n'.format(r * L))
        ofile.write('{0} {1} {2}\n'.format(L, r * L, r * 10**8))
        ofile.close()
        
        haps = list(map(os.path.abspath, sorted(glob.glob(os.path.join(odir, '*.haps')))))
        samples = list(map(os.path.abspath, [u.replace('.haps', '.sample') for u in haps if os.path.exists(u.replace('.haps', '.sample'))]))
        
        # we need to rewrite the haps files (for haploid organisms)
        for sample in samples:
            f = open(sample, 'w')
            f.write('ID_1 ID_2 missing\n')
            f.write('0    0    0\n')
            for k in range(int(args.n_samples)):
                f.write('UNR{} NA 0\n'.format(k + 1))
        
        for ix in range(len(samples)):
            ofile = haps[ix].split('/')[-1].replace('.haps', '') + '_' + map_file.split('/')[-1].replace('.map', '').replace(tag, '').replace('.', '')
            cmd_ = relate_cmd.format(mu, int(args.N), haps[ix], 
                                     samples[ix], map_file, 
                                     ofile, odir)
            
            print(cmd_)
            os.system(cmd_)
            
            f = open(os.path.join(odir, ofile) + '.anc', 'a')
            c_ix = int(re.findall(r'chr\d+', haps[ix].split('/')[-1])[0].replace('chr', ''))
            f.write('chromosome: {}\n'.format(c_ix))
            f.close()
            
            cmd_ = 'cat {} >> {}'.format(os.path.join(odir, ofile) + '.anc', m_ofile)
            print(cmd_)
            os.system(cmd_)
                   
            os.system('rm -rf {}*'.format(os.path.join(odir, ofile)))
        
        os.system('gzip {0}'.format(m_ofile))
        
        os.system('rm -rf {}'.format(os.path.join(odir, '*.sample')))
        os.system('rm -rf {}'.format(os.path.join(odir, '*.haps')))
        
    # compress back
    logging.info('compressing back...')
    ifiles = [os.path.join(idir, u) for u in os.listdir(idir) if (('.ms' in u) or ('.msOut' in u))]
    for ifile in ifiles:
        if '.gz' not in ifile:
            os.system('gzip {}'.format(ifile))

if __name__ == '__main__':
    main()
