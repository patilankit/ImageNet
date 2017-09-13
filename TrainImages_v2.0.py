from keras.applications.vgg16 import VGG16
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Input
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

input_tensor = Input(shape=(64, 64, 3))
model = VGG16(input_tensor = input_tensor,include_top=True, weights=None, pooling=None, classes=1000)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,)


#================================================================================================= Initialize
model = initialize(model= model, X_train=X_train)
#=================================================================================================


#-------------------------------------------------------------------------------------------------- Training
filepath = "/home/ankit/Desktop/DDP/ImageNet/ImageNet_v1.0/BestModels/VGG16/Model - {epoch:02d}-{val_acc:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='val_acc',mode='max', save_best_only=True)
callback_list = [checkpointer]
hist = model.fit(X_train,Y_train, batch_size=32, nb_epoch=5,verbose = 3, validation_split= 0.1,shuffle=True, callbacks= callback_list);      #hist will store everything
#-------------------------------------------------------------------------------------------------- Testing
score = model.evaluate(X_train, Y_train, batch_size=128)
model.save("/home/ankit/Desktop/DDP/ImageNet/ImageNet_v1.0/SavedModels/Model1_VGG_in1.h5")

