from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
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
import os
from PIL import Image
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict
# from tqdm import tqdm
import pickle
import datetime
import dependent_initialization as DI
#
# def rotateImage(image, angle):
#  image_center = tuple(np.array(image[:,:,0].shape)/2)
#  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
#  result = cv2.warpAffine(image, rot_mat, image[:,:,0].shape,flags=cv2.INTER_LINEAR)
#  return result
# '''

########################################### Label Conversion to One Hot Encoding ###############################################
trainPath = '/home/ankit/Desktop/dataset/'
path = os.path.join(trainPath,'trainLabelsCIFAR_Kaggle.csv')
train_df = pd.read_csv(path)
train_labels = pd.get_dummies(train_df['label'])

Y_train = train_labels
print("Read and converted the Training Labels")
################################################################################################################################




def get_array(fname):
    img = load_img(fname,target_size=(128,128));
    x = img_to_array(img);  
    x = x.transpose(2,1,0)
    x = x.reshape((1,3,128,128))
    x = x/255;
    return x;


dr = dict()
path = os.path.join('/media/sai/New Volume1/Practice/statefarm/data', 'driver_imgs_list.csv')
print('Read drivers data')
f = open(path, 'r')
line = f.readline()
while (1):
    line = f.readline()
    if line == '':
        break
    arr = line.strip().split(',')
    dr[arr[2]] = arr[0];
f.close()
#print dr;
unique_drivers = sorted(list(set(dr.values())))
print('Unique drivers: {}'.format(len(unique_drivers)))
print(unique_drivers)
unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p026', 'p035', 'p041', 'p042', 
                     'p045', 'p047', 'p049', 'p050', 'p051', 'p052',  'p061', 'p064', 'p066', 'p072', 'p075', 'p081']
unique_list_valid = ['p056','p039']
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)                     
path = "/media/sai/New Volume1/Practice/statefarm/images/train/c{0}/"
'''
X_train = np.zeros((20925,3,128,128),dtype='float32');
Y_train = np.zeros((20925,1),dtype='int8');
X_valid = np.zeros((1499,3,128,128),dtype='float32');
Y_valid = np.zeros((1499,1),dtype='int8');
j = 0;
k = 0;
'''
for i in tqdm(range(10)):
    newPath = path.format(i);
    if os.path.exists(newPath): 
        #j = j + len(os.listdir(newPath))
        for l in tqdm(os.listdir(newPath)):
            fname =  newPath + l;
            Xtrain = get_array(fname);
            if dr[l] in unique_list_train:
                X_train[k,:,:,:] = Xtrain;
                Y_train[k] = i;
                k = k +1;
            elif dr[l] in unique_list_valid:
                X_valid[j,:,:,:] = Xtrain;
                Y_valid[j] = i;
                j = j +1;

print(j,k);
#X_train = np.asarray(X_train);
#Y_train = np.asarray(Y_train);
#X_valid = np.asarray(X_valid);
#Y_valid = np.asarray(Y_valid);
'''
#X_train = np.load("/media/sai/New Volume1/Practice/statefarm/trainx.npy")
#Y_train = np.load("/media/sai/New Volume1/Practice/statefarm/trainy.npy")
#X_valid = np.load("/media/sai/New Volume1/Practice/statefarm/validx.npy")
#Y_valid = np.load("/media/sai/New Volume1/Practice/statefarm/validy.npy")
print "done extracting data"
print X_train.shape
print X_valid.shape
#'''
Y_train = Y_train.reshape(len(Y_train),1);
Y_valid = Y_valid.reshape(len(Y_valid),1);
Y_train = np_utils.to_categorical(Y_train, 10);
Y_valid = np_utils.to_categorical(Y_valid, 10);
act = 'relu'
# print "creating model"
model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(3,128,128)))
model.add(Convolution2D(32, 3, 3, W_constraint = maxnorm(2)))
model.add(Activation(act))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32, 3, 3, W_constraint = maxnorm(2)))
model.add(Activation(act))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, W_constraint = maxnorm(2)))
model.add(Activation(act))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, W_constraint = maxnorm(2)))
model.add(Activation(act))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, W_constraint = maxnorm(2)))
model.add(Activation(act))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten());
model.add(Dense(4096,W_constraint = maxnorm(2)))
model.add(Activation(act))
model.add(Dropout(0.5))
model.add(Dense(4096,W_constraint = maxnorm(2)))
model.add(Activation(act))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.001, decay=1e-4, momentum=0.5, nesterov=True);
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics =["accuracy"]);
model = DI.initialize(model,"/media/sai/New Volume1/Practice/statefarm/images/train/")
checkpointer = ModelCheckpoint(filepath="/media/sai/New Volume1/Practice/statefarm/model_best/model-{epoch:02d}-{val_loss:.2f}.model", verbose=1, save_best_only=True)
model.fit(X_train,Y_train, batch_size=32, nb_epoch=100,validation_data=(X_valid,Y_valid),shuffle=True,callbacks=[checkpointer]);
model.save_weights('/media/sai/New Volume1/Practice/statefarm/second_try.h5')
#'''



