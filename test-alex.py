# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.optimizers import Momentum
import numpy as np
import pickle

from tflearn.data_utils import image_preloader,build_image_dataset_from_dir
# from sklearn.cross_validation import train_test_split
import os
import tensorflow as tf

from sklearn.metrics import classification_report

split_size = 0.2
learning_rate = 0.005
lr_decay = 0.1
decay_step = 30
momentum = 0.9
weight_decay= 0.0005
num_classes = 8
batch_size = 100

X_test, Y_test = build_image_dataset_from_dir('train/',
                                        dataset_file='output/xray-tmp.pkl',
                                        resize=(227,227),
                                        convert_gray=False,
                                        shuffle_data=False,
                                        categorical_Y=True)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_test.shape,Y_test.shape)


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()


network = input_data(shape=[None, 227, 227, 3],data_preprocessing=img_prep)
network = conv_2d(network, 96, 11, strides=4,weight_decay = weight_decay, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5,weight_decay = weight_decay, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3,weight_decay = weight_decay, activation='relu')
network = conv_2d(network, 384, 3,weight_decay = weight_decay, activation='relu')
network = conv_2d(network, 256, 3,weight_decay = weight_decay, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096,weight_decay = weight_decay ,activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096,weight_decay = weight_decay, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, num_classes,activation='softmax')

momentum = Momentum(learning_rate=learning_rate, staircase=True,lr_decay=lr_decay, decay_step=decay_step)
network = regression(network, optimizer=momentum, loss='categorical_crossentropy', learning_rate=learning_rate)

model = tflearn.DNN(network, checkpoint_path=None,tensorboard_dir='output/tflearn_logs/', tensorboard_verbose=2)

model.load("model/xray-tmp-80-20.tfl")

print("Evaluation")
evaluation= model.evaluate(X_test, Y_test)
print("\n")
print("\t"+"Mean accuracy of the model is :", evaluation)

predict = model.predict(X_test)

predicted_labels = np.argmax(predict,axis=1)

target_labels = np.argmax(Y_test,axis=1)
# target_names = ['Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___healthy', 'Potato___healthy', 'Potato___Late_blight']
target_names = ['blade','bomb','knife']

report = classification_report(target_labels,predicted_labels,target_names=target_names)
print(report)
with open('output/out-data.pkl', 'wb') as out_data:
    pickle.dump([target_labels,predicted_labels], out_data, protocol=2)







