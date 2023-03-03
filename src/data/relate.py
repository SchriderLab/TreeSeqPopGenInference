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

# L = 110000
# mu = 1e-8
# r = 1e-8

def id_generator(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--L", default = "10000")
    parser.add_argument("--mu", default = "1.1e-8")
    parser.add_argument("--r", default = "1.2e-8")
    parser.add_argument("--relate_path", default = "/nas/longleaf/home/ddray/SeqOrSwim/relate/bin/Relate")

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
    for ix in range(len(ifiles)):
        ifile = ifiles[ix]
        
        if '.gz' in ifile:
            os.system('gzip -d {}'.format(ifile))
            
            ifile = ifile.replace('.gz', '')
            
        ifiles[ix] = ifile
    
    rcmd = 'cd {3} && Rscript /nas/longleaf/home/ddray/SeqOrSwim/src/data/ms2haps.R {0} {1} {2}'
    relate_cmd = 'cd {6} && ' + args.relate_path + ' --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'
    
    for ifile in ifiles[:1]:
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
        
        
        """
        ofile = open(ifile.split('.')[0] + '.poplabels', 'w')
        ofile.write('sample population group sex\n')
        for k in range(1, 26):
            ofile.write('UNR{} POP POP 1\n'.format(k))
        ofile.close()
        """

        samples = sorted(glob.glob(os.path.join(odir, '*.sample')))
        haps = sorted(glob.glob(os.path.join(odir, '*.haps')))
        
        for ix in range(len(samples)):
            cmd_ = relate_cmd.format(mu, L, haps[ix], 
                                     samples[ix], map_file, 
                                     haps[ix].split('/')[-1].replace('.haps', ''), odir)
            os.system(cmd_)
        
        #os.system('rm -rf {}'.format(os.path.join(idir, '*.sample')))
        #os.system('rm -rf {}'.format(os.path.join(idir, '*.haps')))
        
    # compress back
    logging.info('compressing back...')
    ifiles = [os.path.join(idir, u) for u in os.listdir(idir) if (('.ms' in u) or ('.msOut' in u))]
    for ifile in ifiles:
        if '.gz' not in ifile:
            os.system('gzip {}'.format(ifile))

if __name__ == '__main__':
    main()
