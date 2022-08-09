# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--classes", default = "mig12,mig21,noMig")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ifiles = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'hdf5'])
    counts = dict()
    
    ofile = h5py.File(args.ofile, 'w')
    ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')

    for ifile in ifiles:
        logging.info('reading {}...'.format(ifile))
        try:
            val = '_val' in ifile
            
            ifile = h5py.File(ifile, 'r')
            
            keys = list(ifile.keys())
            if len(keys) == 0:
                continue
        except:
            continue
        
        cases = sorted(keys)
        
        for case in cases:
            if case not in list(counts.keys()):
                counts[case] = [0, 0]
                
            keys = list(ifile[case].keys())
            
            for key in keys:
                skeys = list(ifile[case][key].keys())
                
                for skey in skeys:
                    if not val:
                        ofile.create_dataset('{1}/{0}/{2}'.format(counts[case][0], case, skey), data = np.array(ifile[case][key][skey]), compression = 'lzf')
                        
                    
                    else:
                        ofile_val.create_dataset('{1}/{0}/{2}'.format(counts[case][1], case, skey), data = np.array(ifile[case][key][skey]), compression = 'lzf')
                    
                if val:
                    counts[case][1] += 1
                else:
                    counts[case][0] += 1

                ofile.flush()
                ofile_val.flush()
                
    ofile.close()
    ofile_val.close()
        
        
        
        
    # ${code_blocks}

if __name__ == '__main__':
    main()

