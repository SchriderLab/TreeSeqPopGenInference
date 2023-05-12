import argparse
from model_viz import cm_analysis, count_parameters

import torch
import torch.nn.functional as F
import h5py
import configparser
from data_loaders import TreeSeqGenerator, TreeSeqGeneratorV2
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn
from gcn_layers import GATSeqClassifier, GATConvClassifier

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel, BCEWithLogitsLoss
from collections import deque

from sklearn.metrics import accuracy_score

import logging, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

"""
example command:
"python3 src/models/train_gcn.py --ifile /pine/scr/d/d/ddray/seln_trees_i2_l128_scattered.hdf5 --ifile_val /pine/scr/d/d/ddray/seln_trees_i2_l128_scattered.hdf5 
    --odir training_results/seln_rnn_i6/ --n_steps 1000 --lr 0.0001 --L 128 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4"
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default="None", help = "training h5 file")
    parser.add_argument("--ifile_val", default="None", help = "validation h5 file")
    parser.add_argument("--odir", default="None")
    
    parser.add_argument("--n_epochs", default="100")
    parser.add_argument("--lr", default="0.00001")
    parser.add_argument("--n_early", default = "10")
    parser.add_argument("--lr_decay", default = "None")
    
    parser.add_argument("--n_per_batch", default = "16")
    parser.add_argument("--L", default = "128", help = "tree sequence length")
    parser.add_argument("--n_steps", default = "3000", help = "number of steps per epoch (if -1 all training examples are run each epoch)")
    
    # data parameter
    parser.add_argument("--in_dim", default = "4")
    parser.add_argument("--n_classes", default = "3")
    
    # hyper-parameters
    parser.add_argument("--use_conv", action = "store_true")
    parser.add_argument("--hidden_dim", default = "128")
    parser.add_argument("--n_gru_layers", default = "1")
    parser.add_argument("--n_gcn_iter", default = "6")
    parser.add_argument("--gcn_dim", default = "26")
    parser.add_argument("--conv_dim", default = "4")
    
    parser.add_argument("--pad_l", action = "store_true")
    
    parser.add_argument("--weights", default = "None")
    parser.add_argument("--weight_decay", default = "0.0")
    parser.add_argument("--momenta_dir", default = "None")
    parser.add_argument("--save_momenta_every", default = "250")
    parser.add_argument("--label_smoothing", default = "0.0")
    parser.add_argument("--regression", action = "store_true")
    
    parser.add_argument("--means", default = "None")
    
    parser.add_argument("--model", default = "gru")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.info('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))
            
    if args.momenta_dir != "None":
        if not os.path.exists(args.momenta_dir):
            os.system('mkdir -p {}'.format(args.momenta_dir))
            logging.info('root: made output directory {0}'.format(args.momenta_dir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.momenta_dir, '*')))

    return args


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    # model = Classifier(config)

    L = int(args.L)

    generator = TreeSeqGeneratorV2(h5py.File(args.ifile, 'r'), means = args.means, n_samples_per = int(args.n_per_batch))
    validation_generator = TreeSeqGeneratorV2(h5py.File(args.ifile_val, 'r'), means = args.means, n_samples_per = int(args.n_per_batch))
    
    if args.model == 'gru':
        model = GATSeqClassifier(generator.batch_size, n_classes = int(args.n_classes), L = L, 
                             n_gcn_iter = int(args.n_gcn_iter), in_dim = int(args.in_dim), hidden_size = int(args.hidden_dim),
                             use_conv = args.use_conv, num_gru_layers = int(args.n_gru_layers), gcn_dim = int(args.gcn_dim))
    elif args.model == 'conv':
        model = GATConvClassifier(generator.batch_size, n_classes = int(args.n_classes), L = L, 
                             n_gcn_iter = int(args.n_gcn_iter), in_dim = int(args.in_dim), hidden_size = int(args.hidden_dim),
                             gcn_dim = int(args.gcn_dim), conv_dim = int(args.conv_dim))
    
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)
    
    classes = generator.models
    
    model = model.to(device)
    print(model)
    count_parameters(model)
    
    # momenta stuff
    save_momenta_every = int(args.save_momenta_every)
    momenta_count = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay = float(args.weight_decay))
    
    # for writing the training 
    result = dict()
    result['epoch'] = []
    result['loss'] = []
    result['acc'] = []
    result['val_loss'] = []
    result['val_acc'] = []

    losses = deque(maxlen=500)
    accuracies = deque(maxlen=500)
    
    if int(args.n_classes) > 1 and not args.regression:
        criterion = LabelSmoothing(float(args.label_smoothing))
        classification = True
    else:
        criterion = nn.SmoothL1Loss()
        classification = False
        
    #print(criterion)
    if args.lr_decay != "None":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, float(args.lr_decay))
    else:
        lr_scheduler = None
        
    min_val_loss = np.inf
    
    for epoch in range(int(args.n_epochs)):
        model.train()
        
        n_steps = int(args.n_steps)
        for j in range(int(args.n_steps)):
            batch, x1, x2, y = generator[j]
            
            if batch is None:
                break
            
            #print(batch.edge_index.shape, batch.x.shape, batch.edge_index.max())
            batch = batch.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad()

            y_pred = model(batch.x, batch.edge_index, batch.batch, x1, x2)

            loss = criterion(torch.squeeze(y_pred), y)

            if classification:
                y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                y = y.detach().cpu().numpy()
    
                accuracies.append(accuracy_score(y, y_pred))

            losses.append(loss.detach().item())

            loss.backward()

            if args.momenta_dir != "None":
                ret = dict()
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        ret[name] = param.grad.detach().cpu().numpy()
                model.update_momenta(ret)
                
                if (j + 1) % save_momenta_every == 0:
                    np.savez(os.path.join(args.momenta_dir, '{0:06d}.npz'.format(momenta_count)), **model.momenta)
                    momenta_count += 1
                    
            optimizer.step()

            # change back to 100
            if (j + 1) % 25 == 0:
                logging.info("root: Epoch: {}/{}, Step: {}/{}, Loss: {}, Acc: {}".format(epoch+1,
                                                                       args.n_epochs, j + 1, n_steps,
                                                                        np.mean(losses), np.mean(accuracies)))

        generator.on_epoch_end()
        
        train_loss = np.mean(losses)
        train_acc = np.mean(accuracies)
        
        val_losses = []
        val_accs = []

        logging.info('validating...')
        model.eval()
        
        Y = []
        Y_pred = []
        with torch.no_grad():
            for j in range(len(validation_generator)):
                batch, x1, x2, y = validation_generator[j]
                
                if batch is None:
                    break
                
                batch = batch.to(device)
                y = y.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)

                y_pred = model(batch.x, batch.edge_index, batch.batch, x1, x2)

                loss = criterion(torch.squeeze(y_pred), y)
                
                if classification:
                    y_pred = y_pred.detach().cpu().numpy().flatten()
                    y = y.detach().cpu().numpy().flatten()
                    
                    y_pred = np.argmax(y_pred, axis=1)
                    val_accs.append(accuracy_score(y, y_pred))
                else:
                    y_pred = y_pred.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()

                Y.extend(y)
                Y_pred.extend(y_pred)

                val_losses.append(loss.detach().item())
                
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        
        result['epoch'].append(epoch)
        result['val_loss'].append(val_loss)
        result['val_acc'].append(val_acc)
        result['loss'].append(train_loss)
        result['acc'].append(train_acc)
        
        logging.info('root: Epoch {}, Val Loss: {:.3f}, Val Acc: {:.3f}'.format(epoch + 1, val_loss, val_acc))
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print('saving weights...')
            torch.save(model.state_dict(), os.path.join(args.odir, 'best.weights'))
            
            # do this for all the examples:
            if classification:
                cm_analysis(Y, np.round(Y_pred), os.path.join(args.odir, 'confusion_matrix_best.png'), classes)
            
            Y = np.array(Y) * generator.y_std + generator.y_mean
            Y_pred = np.array(Y_pred) * generator.y_std + generator.y_mean
                        
            mses = []
            rs = []
            
            for k in range(Y.shape[1]):
                mses.append(np.mean((Y[:,k] - Y_pred[:,k])**2))
                rs.append(np.corrcoef(Y[:,k], Y_pred[:,k])[0, 1])
                
            print(mses, rs)
            
            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break
        
        validation_generator.on_epoch_end(False)
    
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, 'metric_history.csv'), index = False)
        
        if lr_scheduler is not None:
            logging.info('lr for next epoch: {}'.format(lr_scheduler.get_last_lr()))
            lr_scheduler.step()
        
        
    """
    plt.rc('font', family = 'Helvetica', size = 12)
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(12, 8), dpi=100)
    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('negative ll loss')
    ax.set_title('training loss history')
    
    ax
    """  

if __name__ == "__main__":
    main()