# -*- coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def dump_to_ms(X, positions, dir_count, N, odir):
    classes = list(X.keys())
    odir = os.path.join(odir, '{0:04d}'.format(dir_count))
    
    os.system('mkdir -p {}'.format(odir))
    
    for c in classes:
        Xs = X[c][:N]
        pos = positions[c][:N]
        
        del X[c][:N]
        del positions[c][:N]
        
        ofile_ = os.path.join(os.path.join(odir, c + '.msOut'))
        ofile = open(ofile_, 'w')
        
        for x, p in zip(Xs, pos):
            n_seg = x.shape[0]
            header = '//* \n'
            
            ofile.write(header)
            
            seg_line = 'segsites: {}\n'.format(n_seg)
            ofile.write(seg_line)
            
            pos_line = 'positions: {}\n'.format(' '.join(["{0:.5f}".format(u) for u in p]))
            ofile.write(pos_line)
            
            x = x.T.astype(np.uint8)
            for x_ in x:
                line = ''.join([str(u) for u in x_]) + '\n'
                ofile.write(line)
                
            ofile.write('\n')
            
        ofile.close()
        os.system('gzip {}'.format(ofile_))
            
        
    dir_count += 1
    
    return X, positions, dir_count

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_per_dir", default = "100")
    parser.add_argument("--ifile", default = "/pine/scr/d/d/ddray/seln_data.npz")

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

    classes = ['neutral', 'hard', 'hard-near', 'soft', 'soft-near']

    ifile = np.load(args.ifile, allow_pickle = True, encoding = 'latin1')
    keys = list(ifile.keys())

    x_keys = sorted([u for u in keys if 'xtrain' in u])
    pos_keys = sorted([u for u in keys if 'postrain' in u])
    y_keys = sorted([u for u in keys if 'ytrain' in u])
    
    N = int(args.n_per_dir)
    
    X = dict()
    positions = dict()
    
    for c in classes:
        X[c] = []
        positions[c] = []
    
    dir_count = 0
    
    for ix in range(len(x_keys)):
        print('working on {}...'.format(x_keys[ix]))
        
        x = list(ifile[x_keys[ix]])
        pos = list(ifile[pos_keys[ix]])
        
        y = ifile[y_keys[ix]]
        classes_ = [classes[u] for u in y]
        
        for x_, pos_, c in zip(x, pos, classes_):
            X[c].append(x_)
            positions[c].append(pos_)
            
            if all([len(X[c]) >= N for c in classes]):
                X, positions, dir_count = dump_to_ms(X, positions, dir_count, N, args.odir)
                print('dumping files...')
           

if __name__ == '__main__':
    main()

