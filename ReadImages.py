import numpy as np
import os
import pickle
from keras.utils import np_utils




def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

#
#Input: data_folder is where the data is stored, idx is the index of the training set, img_size is the 64*64 in this case
#Output: X_train, Y_train (categorical) [X_train -> {data_size,3,img_size,img_size},   Y_train -> {data_size,1000}]
#Funciton: Wrapper function for getting directly the X_train and Y_train just by giving the folder path and index number
#

def load_databatch(data_folder, idx, img_size = 64):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']
    x = x[0:200, :]  # to reduce the data to ease the computations
    y = y[0:200]     #comment these 2 lines while the original execution

    x = x / np.float32(255)
    mean_image = mean_image / np.float32(255)

                                                                                                                                                                                                                            # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    #===================================================================================================================
    X_train = x[0:data_size, :, :, :]                   #here you can change the data_size of your training data if wished
    Y_train = y[0:data_size]
    #===================================================================================================================

    Y_train = np_utils.to_categorical(Y_train, 1000)

    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)



    return X_train, Y_train


data_folder = "/home/ankit/Desktop/DDP/ImageNet/Train"
idx = 1
img_size = 64
load_databatch(data_folder= data_folder,idx = idx)


