# -*- coding: utf-8 -*-
import os
import argparse
import logging

from collections import OrderedDict

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

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
    
    ### get haplotype and sample files
    ifiles = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if (('.ms' in u) or ('.msOut' in u))]
    rcmd = 'Rscript src/data/ms2haps.R {0} {1}'
    relate_cmd = 'relate/bin/Relate --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'
    
    for ifile in ifiles:
        cmd_ = rcmd.format(ifile, ifile.split('.')[0])
        os.system(cmd_)
        
        # read the ms file for the mutation rate and number of sites
        msf = open(ifile, 'r')
        lines = msf.readlines()
        l = lines[3].replace('//\tOrderedDict(', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
        
        L = float(l[3])  
        r = float(l[5])
        mu = float(l[7])
        
        n_sites = int(lines[4].split(':')[-1].replace('\n',''))
        
        ofile = open(ifile.split('.')[0] + '.map', 'w')
        ofile.write('pos COMBINED_rate Genetic_Map\n')
        ofile.write('0 {} 0\n'.format(r * L))
        ofile.write('{0} {1} {2}\n'.format(L, r * L, r * 10**8))
        ofile.close()
        
        ofile = open(ifile.split('.')[0] + '.poplabels', 'w')
        ofile.write('sample population group sex\n')
        for k in range(1, 26):
            ofile.write('UNR{} POP POP 1\n'.format(k))
        ofile.close()
        
        #ofile = open(ifile.split('.')[0] + '.relate.anc', 'w')
        #ofile.write('>anc\n')
        #ofile.write(sum(['0' for k in range(n_sites)]) + '\n')
        #ofile.close()
        
        cmd_ = relate_cmd.format(mu, L, ifile.split('.')[0] + '.haps', 
                                 ifile.split('.')[0] + '.sample', ifile.split('.')[0] + '.map', 
                                 ifile.split('/')[-1].split('.')[0])
        os.system(cmd_)
        os.system('mv {0}* {1}'.format(ifile.split('/')[-1].split('.')[0], args.odir))

if __name__ == '__main__':
    main()

