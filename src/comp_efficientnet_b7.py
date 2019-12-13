# author: jussikai, timoh

import os 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Model
import efficientnet.tfkeras

X = np.load("X.npy")
y = np.load("y.npy")

weights = [4,3,1,1,1,1,5,3,2,5,1,4,4,3,2,1,1]
Train_X, test_X, Train_y, test_y= sklearn.model_selection.train_test_split(X,y,test_size=0.2)
del X
del y

################
# EfficientNet #
################

base_model = base_model = efficientnet.tfkeras.EfficientNetB7(weights='imagenet', input_shape = (224,224,3), include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(256, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(17, activation = "sigmoid")(w)

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

model.fit(Train_X,Train_y, epochs=15, batch_size=15,validation_data = (test_X, test_y), class_weight = weights)
model.save('EfficientNetB7_Timo.h5')
print("Saved")

del model
del Train_X
del Train_y
del test_X
del test_y
