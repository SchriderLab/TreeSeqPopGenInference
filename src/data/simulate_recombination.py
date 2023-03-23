# -*- coding: utf-8 -*-
import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def chunks(lst, n):
    _ = []
    
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])
        
    return _

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--chunk_size", default = "1000")

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
    
    cmd = 'sbatch -t 02:00:00 --mem=8g --wrap "{}"'
    lines = open(args.ifile, 'r').readlines()
    
    c = chunks(lines, int(args.chunk_size))
    
    for ix, chunk in enumerate(c):
        cmd_file = os.path.join(args.odir, '{0:05d}.sh'.format(ix))
        w = open(cmd_file, 'w')
        ofile = os.path.join(args.odir, '{0:05d}.msOut'.format(ix))
        
        for c_ in chunk:
            w.write(c_.replace('all.LD.sims.txt', ofile).replace('./ms', 'msdir/ms'))
        w.close()
        
        cmd_ = 'chmod +x {0} && {0} && gzip {1}'.format(cmd_file, ofile)
        cmd_ = cmd.format(cmd_)
        
        print(cmd_)
        os.system(cmd_)
        

    # ${code_blocks}

if __name__ == '__main__':
    main()

