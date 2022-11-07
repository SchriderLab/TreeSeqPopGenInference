import numpy as np
import argparse
from data_loaders import SelectionGenotypeMatrixGenerator,ResnetGenotypeMatrixGenerator
from layers import LexSelectionNet, resnet34,LexNet_EXACT
from model_viz import cm_analysis, count_parameters
import torch
import random
from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import deque
from sklearn.metrics import accuracy_score
import logging, os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

torch.cuda.empty_cache()

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
        target2 = target.type(torch.int64)
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target2.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]   #used to replace keras.to_categorical

def pad_matrix_resnet(x,new_size,axis):
    #expects a genotype matrix (channels,sites,n_individuals,) shaped
    s = x.shape[axis]
    
    if new_size > s:
        x_ = np.zeros((x.shape[0],x.shape[1],new_size-s))
        x = np.concatenate([x,x_],axis=axis)
    elif new_size < s:
        segment = s - new_size
        start = random.randint(0,segment)
        return x[:,start:start+new_size,:]
    return x

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--odir", default="None")
    
    parser.add_argument("--n_epochs", default="3")
    parser.add_argument("--lr", default="0.00001") #original is 0.00001
    parser.add_argument("--n_early", default = "10")
    parser.add_argument("--lr_decay", default = "None")
    
    parser.add_argument("--L", default = "32", help = "tree sequence length")
    parser.add_argument("--n_steps", default = "1000", help = "number of steps per epoch (if -1 all training examples are run each epoch)")
    parser.add_argument("--n_classes", default = "5")
    
    parser.add_argument("--weights", default = "None")

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
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args


def main():
    args = parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_classes = int(args.n_classes)
    u = np.load('final.split.up.seln.big.npz', encoding = 'latin1',allow_pickle=True)

    xtest_, ytest_,postest_ = [u[i][:2000] for i in 'xtest ytest postest'.split()]
    

    postest_ = pad_sequences(postest_, padding='post', value=-1., dtype='float32',  maxlen=5000) #unpack and pad from (2000,) to (2000,5000)
    xtest_ = pad_sequences(xtest_, padding='post',  maxlen=5000) #unpack and pad from (2000,) to (2000,5000,208)
    #xtest_ = pad_matrix_resnet(xtest_,256,axis=2)

    #ytest_ = to_categorical(ytest_,n_classes) #converting (2000,) to (2000,5)
    validation_generator = SelectionGenotypeMatrixGenerator(xtest_,ytest_,postest_) #returns torch tensors of batch_size=64

    epochs = int(args.n_epochs)
    n_steps = int(args.n_steps)

    model =  LexNet_EXACT() #resnet34() 

    model = model.to(device)
    print(model)
    count_parameters(model)

    xtrains, ytrains, postrains = [i for i in u.keys() if "xtrain" in i],[i for i in u.keys() if "ytrain" in i], [i for i in u.keys() if "postrain" in i] 
    xtrains.sort(), ytrains.sort(), postrains.sort()
    training_data = list(zip(*[xtrains, ytrains, postrains])) #following what lex did in his model



    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    result = dict()
    result['epoch'] = []
    result['loss'] = []
    result['acc'] = []
    result['val_loss'] = []
    result['val_acc'] = []

    losses = deque(maxlen=500)
    accuracies = deque(maxlen=500)
    criterion = NLLLoss() #LabelSmoothing()  #CrossEntropyLoss #maybe need to find a way to put NLLLoss() here

    if args.lr_decay != "None":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, float(args.lr_decay))
    else:
        lr_scheduler = None

    min_val_loss = np.inf

    for epoch in range(epochs):
        #model.train()
        dataset = 1
        for xtrainx,ytrainx,postrainx in training_data:  #for each of the 78 different train files
            print (epoch, dataset, xtrainx, ytrainx, postrainx)
            xtrain_, ytrain_,postrain_ = u[xtrainx], u[ytrainx], u[postrainx] #load in corresponding np.array
            #ytrain_ = to_categorical(ytrain_, n_classes) #to (3000,5)
            postrain_ = pad_sequences(postrain_, padding='post', value=-1., dtype='float32',  maxlen=5000) #to (3000,5000)
            xtrain_ = pad_sequences(xtrain_, padding='post',  maxlen=5000) #to (3000,5000,208)
            #xtrain_ = pad_matrix_resnet(xtrain_,256,axis=2)
            

            model.train()
            generator = SelectionGenotypeMatrixGenerator(xtrain_,ytrain_,postrain_)
            for j in range(len(generator)):  #shouldn't have to be # of steps. Lex's uses len of generator int(args.n_steps)
                x,y,pos = generator[j] #load each of approx 31 samples batch_size=64
                
                x = x.to(device,dtype=torch.float)
                y = y.to(device,dtype=torch.int64) #ytrain.shape=(64)
                pos = pos.to(device,dtype=torch.float)

                optimizer.zero_grad()
                y_pred = model([x,pos])  #ypred.shape = (64,)
                #y_pred = nn.functional.log_softmax(y_pred,dim=1)
                #ytrain_acc = np.argmax(ytrain.detach().cpu().numpy(),axis=1) #use for NLLLoss maybe

                loss = criterion(y_pred, y)  #Ask what to compare shape wise, if should use np.argmax or not
                y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                y = y.detach().cpu().numpy()
                #y = np.argmax(y,axis=1)

                accuracies.append(accuracy_score(y, y_pred))

                losses.append(loss.detach().item())

                loss.backward()
                optimizer.step()
                if (j + 1) % 25 == 0:
                    logging.info("root: Epoch: {}/{}, Dataset: {}/{}, Step: {}/{}, Loss: {}, Acc: {}".format(epoch+1,epochs, dataset,len(training_data),j + 1, len(generator),np.mean(losses),np.mean(accuracies)))
            
            generator.on_epoch_end()
            dataset = dataset+1

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
                    xtest,ytest,postest = validation_generator[j]

                    xtest = xtest.to(device,dtype=torch.float)
                    ytest = ytest.to(device,dtype=torch.int64)
                    postest = postest.to(device,dtype=torch.float)

                    y_pred = model([xtest, postest])
                    #y_pred = nn.functional.log_softmax(y_pred,dim=1)

                    loss = criterion(y_pred, ytest)
                    y_pred = y_pred.detach().cpu().numpy()
                    y_pred = np.argmax(y_pred, axis=1)
                    ytest = ytest.detach().cpu().numpy()
                    #ytest = np.argmax(ytest, axis=1)
                

                    Y.extend(ytest)
                    Y_pred.extend(y_pred)

                    val_accs.append(accuracy_score(ytest, y_pred))
                    val_losses.append(loss.detach().item())

            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)

            result['epoch'].append(epoch)
            result['val_loss'].append(val_loss)
            result['val_acc'].append(val_acc)
            result['loss'].append(train_loss)
            result['acc'].append(train_acc)

            logging.info('root: Dataset {}, Val Loss: {:.3f}, Val Acc: {:.3f}'.format(epoch + 1, val_loss, val_acc))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('saving weights...')
                torch.save(model.state_dict(), os.path.join(args.odir, 'best.weights'))
                
                # do this for all the examples:
                cm_analysis(Y, np.round(Y_pred), os.path.join(args.odir, 'confusion_matrix_best.png'), ['HardSweep','Hard-Linked','Soft-Sweep','Soft-Linked','Neutral'])

                #early_count = 0
            #else:
                #early_count += 1

                # early stop criteria
                #if early_count > int(args.n_early):
                #    break
            
            validation_generator.on_epoch_end()

            df = pd.DataFrame(result)
            df.to_csv(os.path.join(args.odir, 'metric_history.csv'), index = False)

            if lr_scheduler is not None:
                logging.info('lr for next epoch: {}'.format(lr_scheduler.get_lr()))
                lr_scheduler.step()

if __name__ == "__main__":
    main()