# -*- coding: utf-8 -*-

import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

TODO_DROS = [('/overflow/dschridelab/ddray/four_problems/dros/mt1_3/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt1_3/n512.hdf5 --out_shape 1,34,512 --mode pad'), 
             ('/overflow/dschridelab/ddray/four_problems/dros/mt1_3/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt1_3/n256.hdf5 --out_shape 1,34,256 --mode pad'),
             ('/overflow/dschridelab/ddray/four_problems/dros/mt2_4/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt2_4/n512.hdf5 --out_shape 1,34,512 --mode pad'), 
             ('/overflow/dschridelab/ddray/four_problems/dros/mt2_4/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt2_4/n256.hdf5 --out_shape 1,34,256 --mode pad'),
             ('/overflow/dschridelab/ddray/four_problems/dros/mt3_5/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt3_5/n512.hdf5 --out_shape 1,34,512 --mode pad'), 
             ('/overflow/dschridelab/ddray/four_problems/dros/mt3_5/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt3_5/n256.hdf5 --out_shape 1,34,256 --mode pad'),
             ('/overflow/dschridelab/ddray/four_problems/dros/mt4_6/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt4_6/n512.hdf5 --out_shape 1,34,512 --mode pad'), 
             ('/overflow/dschridelab/ddray/four_problems/dros/mt4_6/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt4_6/n256.hdf5 --out_shape 1,34,256 --mode pad'),
             ('/overflow/dschridelab/ddray/four_problems/dros/mt1_3/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt1_3/n512_cosine.hdf5 --out_shape 2,32,512 --mode seriate_match'), 
            ('/overflow/dschridelab/ddray/four_problems/dros/mt1_3/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt1_3/n256_cosine.hdf5 --out_shape 2,32,256 --mode seriate_match'),
            ('/overflow/dschridelab/ddray/four_problems/dros/mt2_4/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt2_4/n512_cosine.hdf5 --out_shape 2,32,512 --mode seriate_match'), 
            ('/overflow/dschridelab/ddray/four_problems/dros/mt2_4/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt2_4/n256_cosine.hdf5 --out_shape 2,32,256 --mode seriate_match'),
            ('/overflow/dschridelab/ddray/four_problems/dros/mt3_5/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt3_5/n512_cosine.hdf5 --out_shape 2,32,512 --mode seriate_match'), 
            ('/overflow/dschridelab/ddray/four_problems/dros/mt3_5/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt3_5/n256_cosine.hdf5 --out_shape 2,32,256 --mode seriate_match'),
            ('/overflow/dschridelab/ddray/four_problems/dros/mt4_6/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt4_6/n512_cosine.hdf5 --out_shape 2,32,512 --mode seriate_match'), 
            ('/overflow/dschridelab/ddray/four_problems/dros/mt4_6/train_val_sims/', '--ofile /overflow/dschridelab/ddray/four_problems/dros/mt4_6/n256_cosine.hdf5 --out_shape 2,32,256 --mode seriate_match')
             ]

TODO_SELECTION = [('/overflow/dschridelab/ddray/four_problems/selection/sims/train_val', '--ofile /overflow/dschridelab/ddray/four_problems/selection/n2048.hdf5 --out_shape 1,104,2048 --mode pad --pop_sizes 104,0'),
                  ('/overflow/dschridelab/ddray/four_problems/selection/sims/train_val', '--ofile /overflow/dschridelab/ddray/four_problems/selection/n1024.hdf5 --out_shape 1,104,1024 --mode pad --pop_sizes 104,0'),
                  ('/overflow/dschridelab/ddray/four_problems/selection/sims/train_val', '--ofile /overflow/dschridelab/ddray/four_problems/selection/n1024_cosine.hdf5 --out_shape 1,104,1024 --mode seriate --pop_sizes 104,0'),
                  ('/overflow/dschridelab/ddray/four_problems/selection/sims/train_val', '--ofile /overflow/dschridelab/ddray/four_problems/selection/n2048_cosine.hdf5 --out_shape 1,104,2048 --mode seriate --pop_sizes 104,0')]


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--problem", default = "dros")

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
    
    cmd = 'sbatch -t 5-00:00:00 --mem=16G -n 24 --wrap "mpirun -n 24 python3 src/data/format_genomat.py --idir {} {}"'
    
    if args.problem == 'dros':
        for u, v in TODO_DROS:
            cmd_ = cmd.format(u, v)
            print(cmd_)
    elif args.problem == 'selection':
        for u, v in TODO_SELECTION:
            cmd_ = cmd.format(u, v)
            print(cmd_)
        
    

    # ${code_blocks}

if __name__ == '__main__':
    main()