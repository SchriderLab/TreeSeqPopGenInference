# -*- coding: utf-8 -*-
import os
import argparse
import logging

from collections import OrderedDict
import glob

from data_functions import load_data

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--L", default = "10000")
    parser.add_argument("--mu", default = "3.5e-9")
    parser.add_argument("--r", default = "1.75e-8")

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
    
    idir = args.idir
    odir = args.odir
    
    logging.info('working on {}...'.format(idir))
    
    ### get haplotype and sample files
    logging.info('uncompressing data...')
    ifiles = [os.path.join(idir, u) for u in os.listdir(idir) if (('.ms' in u) or ('.msOut' in u))]
    for ifile in ifiles:
        if '.gz' in ifile:
            os.system('gzip -d {}'.format(ifile))
    
    rcmd = 'Rscript src/data/ms2haps.R {0} {1} {2}'
    relate_cmd = 'relate/bin/Relate --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'
    
    for ifile in ifiles:
        ifile = ifile.replace('.gz', '')
        
        tag = ifile.split('/')[-1].split('_')[0]
        
        logging.info('working on {}...'.format(ifile))
        logging.info('converting to haps / sample files via Rscript...')
        cmd_ = rcmd.format(ifile, ifile.split('.')[0], int(args.L))
        os.system(cmd_)
        
        # read the ms file for the mutation rate and number of sites
        msf = open(ifile, 'r')
        lines = msf.readlines()
        
        L = float(args.L)  
        r = float(args.r)
        mu = float(args.mu)
        
        ofile = open(ifile.split('.')[-2] + '.map', 'w')
        ofile.write('pos COMBINED_rate Genetic_Map\n')
        ofile.write('0 {} 0\n'.format(r * L))
        ofile.write('{0} {1} {2}\n'.format(L, r * L, r * 10**8))
        ofile.close()
        
        ofile = open(ifile.split('.')[-2] + '.poplabels', 'w')
        ofile.write('sample population group sex\n')
        for k in range(1, 26):
            ofile.write('UNR{} POP POP 1\n'.format(k))
        ofile.close()
        
        #ofile = open(ifile.split('.')[0] + '.relate.anc', 'w')
        #ofile.write('>anc\n')
        #ofile.write(sum(['0' for k in range(n_sites)]) + '\n')
        #ofile.close()
        
        map_file = ifile.split('.')[-2] + '.map'
        samples = sorted(glob.glob(os.path.join(idir, '*.sample')))
        haps = sorted(glob.glob(os.path.join(idir, '*.haps')))
        
        for ix in range(len(samples)):
            cmd_ = relate_cmd.format(mu, L, haps[ix], 
                                     samples[ix], map_file, 
                                     haps[ix].split('/')[-1].split('.')[0])
            os.system(cmd_)
        
            os.system('mv {0}.anc {1}'.format(haps[ix].split('/')[-1].split('.')[0], odir))
            os.system('mv {0}.mut {1}'.format(haps[ix].split('/')[-1].split('.')[0], odir))
            os.system('rm -rf {}*'.format(haps[ix].split('/')[-1].split('.')[0]))
        
        os.system('rm -rf {}'.format(os.path.join(idir, '*.sample')))
        os.system('rm -rf {}'.format(os.path.join(idir, '*.haps')))
        os.system('rm -rf {}*'.format(tag))
        
    # compress back
    logging.info('compressing back...')
    ifiles = [os.path.join(idir, u) for u in os.listdir(idir) if (('.ms' in u) or ('.msOut' in u))]
    for ifile in ifiles:
        if '.gz' not in ifile:
            os.system('gzip {}'.format(ifile))

if __name__ == '__main__':
    main()

