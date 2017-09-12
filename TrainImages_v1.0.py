from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout,GaussianNoise
from keras.optimizers import SGD,RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import numpy as np
# from tqdm import tqdm
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier
from keras.constraints import maxnorm
# import cv2
import random
import os
from PIL import Image
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict
# from tqdm import tqdm
import pickle
import datetime
from dependant_initialization_v1 import initialize
from ReadImages import load_databatch

data_folder = "/home/ankit/Desktop/DDP/ImageNet/Train"
idx = 1
img_size = 64
# # load_databatch(data_folder= data_folder,idx = idx)
#
#
X_train,Y_train = load_databatch(data_folder = data_folder, idx=idx, img_size= img_size)

# X_train = np.load('X_train.npy')


act = 'relu'
# print "creating model"
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), data_format= "channels_last"))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


model2 = initialize(model,X_train)
