# # from __future__ import print_function
#
# import os
# from keras.models import Model
# import numpy as np;
# import random
#
#
# # dataPath = "/home/ankit/Desktop/dataset/CIFAR_Kaggle/train"
#
#
# def initialize(model, X_train):
#     number = 0; number2 = 0; number3 = 0; number4 = 0;
#     # configfiles = [os.path.join(dirpath, f)
#     #                for dirpath, dirnames, files in os.walk(dataPath)
#     #                for f in files if f.endswith('.png')]  # list all the files with a given extension in directory
#     totalFiles = list(np.arange(X_train.shape[0]))
#     # configfiles = configfiles[:100]  # reduce the file size easy handling
#     random.shuffle(totalFiles);
#     layers = model.layers;
#     for layer in layers:
#         if len(layer.get_weights()) != 0:
#             sample = random.sample(totalFiles, 32);
#             sample = X_train[sample]    #now take the only 32 X_train values after taking random on indexes
#             # gaussian initialization
#             row, col, channels, filters = layer.get_weights()[0].shape;
#             print(number); number += 1;
#             weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
#             weights1 = np.zeros(filters, dtype="float32");
#             mean = np.zeros(row * col, dtype="float32");
#             cov = np.identity(row * col, dtype="float32");
#             for i in range(filters):
#                 for j in range(channels):
#                     weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
#             layer.set_weights([weights0, weights1]);
#             for img in sample:
#                 intermediate_layer_model = Model(inputs=model.input,
#                                                  outputs=layer.output)
#                 intermediate_output = intermediate_layer_model.predict(img.reshape(1,64,64,3));
#                 # intermediate_output = intermediate_layer_model.predict(img);
#                 avg_img = np.average(intermediate_output, axis=1);
#                 for i in range(filters):
#                     # avg = avg_img[ :, :, i].mean;
#                     # std = avg_img[ :, :, i].std;
#                     avg = np.average(avg_img[ :, :, i]);
#                     std = np.std(avg_img[ :, :, i]);
#                     mean = np.ones(row * col, dtype="float32") * avg;
#                     cov = np.identity(row * col, dtype="float32") * (std ** 2);
#                     for j in range(channels):
#                         weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
#                     weights1[i] = avg;
#             layer.set_weights([weights0, weights1]);
#     return model;
#

# model2 = initialize(model= model, X_train=X_train)
#


import os
from keras.models import Model
import numpy as np;
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.preprocessing import normalize

def get_array(fname):
    img = load_img(fname, target_size=(128, 128));
    x = img_to_array(img);
    x = x.transpose(2, 1, 0)
    x = x.reshape((3, 128, 128))
    x = x / 255;
    return x;


def initialize(model, X_train):
    # configfiles = [os.path.join(dirpath, f)
    #                for dirpath, dirnames, files in os.walk(dataPath)
    #                for f in files if f.endswith('.jpg')]  # list all the files with a given extension in directory
    # random.shuffle(configfiles);

    totalFiles = list(np.arange(X_train.shape[0]))
    # configfiles = configfiles[:100]  # reduce the file size easy handling
    random.shuffle(totalFiles);

    layers = model.layers;
    for num, layer in enumerate(layers):
        if len(layer.get_weights()) != 0:
#----------------------------------------------------------------------------------------- Convolution Layers
            if len(layer.get_weights()[0].shape) == 4:
                # sample = random.sample(configfiles, 8);
                sample = random.sample(totalFiles, 8);
                sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes

                # data = np.zeros((8, 3, 64, 64));
                data = np.zeros((8, 64, 64, 3));            #channels last

                # for index, img in enumerate(sample):
                #     data[index, :, :, :] = get_array(img);
                data = sample
                # gaussian initialization
                row, col, channels, filters = layer.get_weights()[0].shape;
                weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
                weights1 = np.zeros(filters, dtype="float32");
                mean = np.zeros(row * col, dtype="float32");
                cov = np.identity(row * col, dtype="float32");
                for i in range(filters):
                    for j in range(channels):
                        weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
                layer.set_weights([weights0, weights1]);

                for img in sample:
                    intermediate_layer_model = Model(inputs=model.input,
                                                     outputs=layer.output)
                    # intermediate_output = np.float32(1.0)
                    intermediate_output = intermediate_layer_model.predict(data);
                    intermediate_output = intermediate_output/max(intermediate_output.flatten())           #random approximation
                    avg_img = np.average(intermediate_output, axis=0);
                    for i in range(filters):
                        avg = avg_img[ :, :, i].mean();         #both are filters last
                        std = avg_img[ :, :, i].std(); #print("Std =",std)
                        mean = np.ones(row * col, dtype="float32") * avg;
                        cov = np.identity(row * col, dtype="float32") * (std ** 2); #print("Cov is =", cov)
                        for j in range(channels):
                            weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
                        weights1[i] = avg;
                layer.set_weights([weights0, weights1]);
                print("done initializing layer " + str(num));
#--------------------------------------------------------------------------------------- Dense Layers

            elif len(layer.get_weights()[0].shape) == 2:
                sample = random.sample(totalFiles, 8);
                sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes

                data = np.zeros((8, 64, 64, 3));            #channels last
                data = sample
                weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
                weights1 = np.zeros(layer.get_weights()[1].shape, dtype="float32");
#-------------------------------------------------------------------------------------------------- Intermediate Model
                intermediate_layer_model = Model(inputs=model.input,outputs=layer.output)
                intermediate_output = intermediate_layer_model.predict(data);
                intermediate_output = intermediate_output / max(intermediate_output.flatten())  # random approximation

                avg_img = np.average(intermediate_output, axis=0);
                if avg_img.std() == (float('inf') or float('nan')) : std = (avg_img/avg_img[0]).std();
                else: std = avg_img.std();

                cov = np.identity(len(avg_img)) * (std ** 2);

#-------------------------------------------------------------------------------------------------- Weights Initialization
                weights1 = avg_img              #bias are initialized for average of the samples
                weights0 = np.random.multivariate_normal(mean = avg_img,cov= cov,size=(layer.get_weights()[0].shape[0]))
                layer.set_weights([weights0, weights1]);
                print("done initialing layer " + str(num));
    return model;

#
#
# model2 = initialize(model= model, X_train=X_train)
#
#
#
