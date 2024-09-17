from __future__ import division, print_function, absolute_import

import PIL

from skimage import color, io
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

import decimal
from six.moves import cPickle
import pickle

import h5py

np.set_printoptions(suppress=True)

# Imports picture file into the model

# TumorA = astrocytoma = 0
# TumorB = glioblastoma_multiforme = 1
# TumorC = oligodendroglioma = 2
# healthy = 3
# unknown = 4

f = open('full_dataset_final.pkl', 'rb')
print("pickle file open")

# Load from the file for X(image data) and Y(tumor type)
allX, allY = pickle.load(f)
print("pickle opened")
f.close()

size_image = 64


# Model architecture

network = input_data(shape=[None, size_image, size_image, 3])

conv_1 = conv_2d(network, nb_filter=64, filter_size=3, activation='relu', name='conv_1')
print("layer 1")

network = max_pool_2d(conv_1, 2)
print("layer 2")

conv_11 = conv_2d(network, nb_filter=96, filter_size=3, activation='relu', name='conv_11')
print("layer 3")

network = max_pool_2d(conv_11, 2)
print("layer 4")

conv_2 = conv_2d(conv_11, nb_filter=64, filter_size=4, activation='relu', name='conv_2')
print("layer 5")

network = max_pool_2d(conv_2, 2)
print("layer 6")

conv_22 = conv_2d(conv_2, nb_filter=96, filter_size=4, activation='relu', name='conv_22')
print("layer 7")

network = max_pool_2d(conv_22, 2)
print("layer 8")

conv_3 = conv_2d(conv_22, nb_filter=64, filter_size=5, activation='relu', name='conv_3')
print("layer 9")

network = max_pool_2d(conv_3, 2)
print("layer 10")

conv_33 = conv_2d(conv_3, nb_filter=96, filter_size=5, activation='relu', name='conv_33')
print("layer 11")

network = max_pool_2d(conv_33, 2)
print("layer 12")

conv_4 = conv_2d(conv_33, nb_filter=64, filter_size=6, activation='relu', name='conv_4')
print("layer 13")

network = max_pool_2d(conv_4, 2)
print("layer 14")

conv_44 = conv_2d(conv_4, nb_filter=96, filter_size=6, activation='relu', name='conv_44')
print("layer 15")

network = max_pool_2d(conv_44, 2)
print("layer 16")

network = fully_connected(network, 512, activation='relu')
print("layer 17")

# 8: Dropout layer to combat overfitting
network = dropout(network, 0.5)
print("layer 18")

# 9: Fully-connected layer with 5 outputs for five tumor categories
network = fully_connected(network, 5, activation='softmax')
print("layer 19")

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose = 0)

print("model created done")

# Train Model

no_folds = 2 # 6 fold cross validation

accuracy_array = np.zeros((no_folds), dtype='float64') 
accuracy_array2 = np.zeros((no_folds), dtype='float64') 

i=0 
split_no = 1 

kf = KFold(n_splits=no_folds, shuffle = True, random_state=42) 

for train_index, test_index in kf.split(allX):

    # split dataset 
    X, X_test = allX[train_index], allX[test_index]
    Y, Y_test = allY[train_index], allY[test_index]

    # Output labels for whole dataset and test dataset
    Y = to_categorical(Y, 5)
    Y_test = to_categorical(Y_test, 5)

    print("train split: " , split_no)
    split_no += 1 

    model.fit(X, Y, n_epoch=1, run_id='cancer_detector', shuffle=True,
        show_metric=True)

    print("Network trained")

    # Calculate accuracies
    score = model.evaluate(X_test, Y_test)
    score2 = model.evaluate(X, Y)

    # populate the accuracy arrays
    accuracy_array[i] = score[0] * 100
    accuracy_array2[i] = score2[0] * 100
    i += 1 

    print("accuracy checked")
    print("")
    print("accuracy for test dataset: ", accuracy_array) 
    print("")
    print("accuracy for whole dataset: ", accuracy_array2) 


print("done training using 6 fold validation")

max_accuracy = accuracy_array[np.argmax(accuracy_array)]
max_accuracy = round(max_accuracy, 3)

max_accuracy2 = accuracy_array2[np.argmax(accuracy_array2)]
max_accuracy2 = round(max_accuracy2, 3)

print("")

# Test the model 

y_label = 0

j = 0
k = 0
c = 0
b = 0

y_pred = np.zeros((len(allY)), dtype='int32')
y_true = np.zeros((len(allY)), dtype='int32')

x_list = np.array_split(allX, 90)
y_list = np.array_split(allY, 90)

i = 0

for j in x_list:
    
    x_test = x_list[i]
    y_test = y_list[i]

    
    y_label = model.predict(x_test)
    print("running here")

    b = 0 
    for k in y_label:
        y_pred[c] = np.argmax(y_label[b]) 
        y_true[c] = y_test[b] 
        c += 1
        b += 1
    i += 1

print("Prediction finished", c)
print("")
print(len(y_true), " ", len(y_pred))
print("")

# calculates F1-Score
print("calculate f1 score")
f1Score = f1_score(y_true, y_pred, average=None)
print(f1Score)

print("")

# calculates Confusion Matrix 
print("calculate confusion matrix")
confusionMatrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
print("confusion Matrix Created")
print(confusionMatrix)


#Result

print("")
print("")
print ("-----------------------------------------------------------------------------")
print ( "    Cancer Tumor detector using Convolutional Neural Networks    ") 
print ("-----------------------------------------------------------------------------")
print("")
print("accuracy for the test dataset")
print(accuracy_array)
print("")
print("accuracy for the whole dataset")
print(accuracy_array2)
print("")
print("Maximum accuracy for test dataset: ", max_accuracy, '%')
print("")
print("Maximum accuracy for whole dataset: ", max_accuracy2, '%')
print("")
print("F1 score for the whole dataset")
print(f1Score)
print("")
print("confusion Matrix")
print(confusionMatrix)
print("")
print ("-----------------------------------------------------------------------------")

