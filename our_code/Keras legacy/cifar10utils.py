from __future__ import print_function, division

from keras.models import Sequential, Model
from keras.models import Sequential
from keras import backend as K
from keras.datasets import cifar10
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import keras

'''
Function that returns the trainand test data of the CIFAR10 already preprocessed
'''
def getCIFAR10():
    # input image dimensions
    img_rows, img_cols = 32, 32
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # format of the tensor
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    # convert in to float the images
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # new normalization with z-score
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print('CIFAR10 loaded')
    return x_train,y_train,x_test,y_test

'''
Small function that returns the shape of the CIFAR10 images
'''
def getCIFAR10InputShape():
    img_rows, img_cols = 32, 32
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)
        
    return input_shape