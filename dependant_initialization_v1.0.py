import os
from keras.models import Model
import numpy as np;
import random


# dataPath = "/home/ankit/Desktop/dataset/CIFAR_Kaggle/train"


def initialize(model, X_train):
    # configfiles = [os.path.join(dirpath, f)
    #                for dirpath, dirnames, files in os.walk(dataPath)
    #                for f in files if f.endswith('.png')]  # list all the files with a given extension in directory
    totalFiles = list(np.arange(X_train.shape[0]))
    # configfiles = configfiles[:100]  # reduce the file size easy handling
    random.shuffle(totalFiles);
    layers = model.layers;
    for layer in layers:
        if len(layer.get_weights()) != 0:
            sample = random.sample(totalFiles, 32);
            sample = X_train[sample]    #now take the only 32 X_train values after taking random on indexes
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
                intermediate_output = intermediate_layer_model.predict(img.reshape(1,64,64,3));
                # intermediate_output = intermediate_layer_model.predict(img);
                avg_img = np.average(intermediate_output, axis=1);
                for i in range(filters):
                    # avg = avg_img[ :, :, i].mean;
                    # std = avg_img[ :, :, i].std;
                    avg = np.average(avg_img[ :, :, i]);
                    std = np.std(avg_img[ :, :, i]);
                    mean = np.ones(row * col, dtype="float32") * avg;
                    cov = np.identity(row * col, dtype="float32") * std ** 2;
                    for j in range(channels):
                        weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
                    weights1[i] = avg;
            layer.set_weights([weights0, weights1]);
    return model;


model2 = initialize(model= model, X_train=X_train)
