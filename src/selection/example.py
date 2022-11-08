import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--odir", default="None")
    
    parser.add_argument("--n_epochs", default="3")
    parser.add_argument("--lr", default="0.00001") #original is 0.00001
    parser.add_argument("--n_early", default = "10")
    parser.add_argument("--lr_decay", default = "None")
    
    parser.add_argument("--L", default = "32", help = "tree sequence length")
    parser.add_argument("--n_steps", default = "3000", help = "number of steps per epoch (if -1 all training examples are run each epoch)")
    parser.add_argument("--n_classes", default = "5")
    
    parser.add_argument("--weights", default = "None")

    args = parser.parse_args()

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()
    print(args)

main()
