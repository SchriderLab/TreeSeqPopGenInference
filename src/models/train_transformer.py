# -*- coding: utf-8 -*-
import argparse
from model_viz import cm_analysis, count_parameters

import torch
import torch.nn.functional as F
import h5py
import configparser
from data_loaders import ProjGenerator
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn
from layers import TransformerClassifier

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel, BCEWithLogitsLoss
from collections import deque

from sklearn.metrics import accuracy_score

import logging, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default="None", help = "training h5 file")
    parser.add_argument("--ifile_val", default="None", help = "validation h5 file")
    parser.add_argument("--odir", default="None")
    
    parser.add_argument("--n_epochs", default="100")
    parser.add_argument("--lr", default="0.0001")
    parser.add_argument("--n_early", default = "10")
    parser.add_argument("--lr_decay", default = "None")
    parser.add_argument("--weight_decay", default = "0.0")
    
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    parser.add_argument("--lambda_ortho", default = "0.1")
    parser.add_argument("--n_per", default = "16")
    
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
            
    return args

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    
    n_per = int(args.n_per)
    
    generator = ProjGenerator(h5py.File(args.ifile), args.ifile.replace('.hdf5', '.npz'), n_per = n_per)
    validation_generator = ProjGenerator(h5py.File(args.ifile_val), args.ifile.replace('.hdf5', '.npz'), n_per = n_per)
    
    model = TransformerClassifier()
    model = model.to(device)
    print(model)
    count_parameters(model)
    
    classes = args.classes.split(',')

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay = float(args.weight_decay))
    
    weights = torch.FloatTensor((generator.counts / np.sum(generator.counts)) ** -1).to(device)
    
    losses = deque(maxlen=500)
    accuracies = deque(maxlen=500)
    criterion = torch.nn.NLLLoss(weight = weights)
    ortho_criterion = OrthogonalProjectionLoss()
    
    

    # for writing the training 
    result = dict()
    result['epoch'] = []
    result['loss'] = []
    result['acc'] = []
    result['val_loss'] = []
    result['val_acc'] = []
    
    min_val_loss = np.inf
    
    for epoch in range(int(args.n_epochs)):
        model.train()
        
        for j in range(len(generator)):
            x, x1, x2, y = generator[j]
            
            #print(y)
            
            #print(batch.edge_index.shape, batch.x.shape, batch.edge_index.max())
            x = x.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad()

            y_pred, f = model(x, x1, x2)

            loss = criterion(y_pred, y) + float(args.lambda_ortho) * ortho_criterion(f, y)

            y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y = y.detach().cpu().numpy()

            accuracies.append(accuracy_score(y, y_pred))

            losses.append(loss.detach().item())

            loss.backward()
            optimizer.step()

            # change back to 100
            if (j + 1) % 25 == 0:
                logging.info("root: Epoch: {}/{}, Step: {}/{}, Loss: {}, Acc: {}".format(epoch+1,
                                                                       args.n_epochs, j + 1, len(generator),
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
                x, x1, x2, y = validation_generator[j]
                
                x = x.to(device)
                y = y.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)

                y_pred, f = model(x, x1, x2)

                loss = criterion(y_pred, y) + float(args.lambda_ortho) * ortho_criterion(f, y)

                y_pred = y_pred.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                
                Y.extend(y)
                Y_pred.extend(y_pred)

                val_accs.append(accuracy_score(y, y_pred))
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
            cm_analysis(Y, np.round(Y_pred), os.path.join(args.odir, 'confusion_matrix_best.png'), classes)

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break
        
        validation_generator.on_epoch_end(False)
    
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, 'metric_history.csv'), index = False)
       
if __name__ == '__main__':
    main()
