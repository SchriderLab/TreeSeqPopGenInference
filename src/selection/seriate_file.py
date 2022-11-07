import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from seriate import seriate
from scipy.spatial.distance import pdist,squareform,cdist
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
#from scipy.optimize import linear_sum_assignment 


def seriate_xpos(x,pos,metric = 'cosine'):
    Dx = pdist(x, metric = metric)
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx, timeout = 0)

    return x[ix],pos[ix],ix

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

# def seriate_(X_,pos):

#     X_sorted,pos_sorted,X_sorted_indicies = seriate_x_pos(X_,pos)
#     #X_sorted_pop2,X_sorted_indicies = seriate_x(X_pop2)     

#     #X_sorted = np.concatenate((X_sorted_pop1,X_sorted_pop2))
#     return X_sorted,pos_sorted

def worker(args):
    sample, pos, idx = args #Unpack the packed tuples
    return seriate_xpos(sample,pos),idx

u = np.load('final.split.up.seln.big.npz', encoding = 'latin1',allow_pickle=True)
pool = mp.Pool(mp.cpu_count()) #Max number of CPUs given to program (or an arbitrary int if desired)

xtest_, ytest_,postest_ = [u[i][:2000] for i in 'xtest ytest postest'.split()]

postest_ = pad_sequences(postest_, padding='post', value=-1., dtype='float32',  maxlen=5000) #unpack and pad from (2000,) to (2000,5000)
xtest_ = pad_sequences(xtest_, padding='post',  maxlen=5000) #unpack and pad from (2000,) to (2000,5000,208)

xtrains, ytrains, postrains = [i for i in u.keys() if "xtrain" in i],[i for i in u.keys() if "ytrain" in i], [i for i in u.keys() if "postrain" in i] 
xtrains.sort(), ytrains.sort(), postrains.sort()
training_data = list(zip(*[xtrains, ytrains, postrains])) #following what lex did in his model

x_train_list = []
y_train_list = []
pos_train_list = []
for xtrainx,ytrainx,postrainx in training_data:  #for each of the 78 different train files
    print (xtrainx, ytrainx, postrainx)
    xtrain_, ytrain_,postrain_ = u[xtrainx], u[ytrainx], u[postrainx] #load in corresponding np.array
    np_x_out = np.zeros((xtrain_.shape[0],5000,256))
    np_pos_out = np.zeros((xtrain_.shape[0],5000))
    #ytrain_ = to_categorical(ytrain_, n_classes) #to (3000,5)
    postrain_ = pad_sequences(postrain_, padding='post', value=-1., dtype='float32',  maxlen=5000) #to (3000,5000)
    xtrain_ = pad_sequences(xtrain_, padding='post',  maxlen=5000) #to (3000,5000,208)
    xtrain_ = pad_matrix_resnet(xtrain_,256,axis=2)
    #postrain_ = pad_matrix_resnet(postrain_,256,axis=2)
    for i in tqdm(range(1,xtrain_.shape[0],100)):
        #xtrain_sorted,ytrain_sorted = seriate_(xtrain_[i,:,:],postrain_[i,:])
        batch = [] #Tuples of (sample, id)
        for j in range(i, i+100): #Iterate through each idx in batch and store with raw data
            batch.append((np.array(xtrain_[j-1,:,:]),np.array(postrain_[j-1,:]), j))
        results = pool.map(worker, batch, chunksize=4) #Now we can throw into parallel processing, the most time intensive part
        for res in results:
            np_x_out[i,:,:] = res[0][0]
            np_pos_out[i,:] = res[0][1]
    
    # if round(xtrain_.shape[0],-2) >= xtrain_.shape[0]:
    #     num_left = round(xtrain_.shape[0],-2) - 99
    # else:
    #     num_left = round(xtrain_.shape[0],-2) + 1 
    # print(num_left)

    # for i in tqdm(range(num_left, xtrain_.shape[0]), desc=f"Seriating data in"):  #for last 100 samples that cant be pooled
    #     x_ = np.array(xtrain_[j,:,:])
    #     pos_ = np.array(postrain_[j,:])
    #     x_sorted,pos_sorted,ix = seriate_(x_,pos_)
    #     np_x_out[i,:,:] = x_sorted
    #     np_pos_out[i,:] = pos_sorted

    x_ = np.array(xtrain_[0,:,:])
    pos_ = np.array(postrain_[0,:])
    x_sorted,pos_sorted,ix = seriate_xpos(x_,pos_)
    np_x_out[0,:,:] = x_sorted
    np_pos_out[0,:] = pos_sorted

    x_train_list.append(np_x_out)
    y_train_list.append(ytrain_)
    pos_train_list.append(np_pos_out)


x_train = np.vstack(x_train_list)
y_train = np.vstack(y_train_list)
pos_train = np.vstack(pos_train_list)
    

np.savez_compressed('seriated_selection.npz', xtrain=x_train, ytrain=y_train,postrain=pos_train)


