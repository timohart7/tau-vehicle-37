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

directory = "data/train/train"
class_names = sorted(os.listdir(directory))

print(class_names)

# Find all image files in the data directory.

X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
#bias = np.zeros((128,128,1))

for root, dirs, files in os.walk(directory):
    #print(dirs)
    for name in files:
        #print(files)
        if name.endswith('.jpg'):
            path = os.path.join(root, name)
            print(path)
            # Load the image:
            img = plt.imread(root + os.sep + name)
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32)
            img -= 128
            #img = np.concatenate((img,bias),axis=2)

            # Convert the data to float, and remove mean          
            # And append the feature vector to our list.
            X.append(img)
            print(name)
            print(os.sep)
            # Extract class name from the directory name:
            label = path.split(os.sep)[-2]
            print(label)
            y.append(class_names.index(label))
            print(name)

X = np.array(X)
y = tf.keras.utils.to_categorical(np.array(y))

weights = [4,3,1,1,1,1,5,3,2,5,1,4,4,3,2,1,1]
Train_X, test_X, Train_y, test_y= sklearn.model_selection.train_test_split(X,y,test_size=0.2)
del X
del y

################
# EfficientNet #
################

base_model = base_model = efficientnet.tfkeras.EfficientNetB7(weights='imagenet', include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(256, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(17, activation = "sigmoid")(w)

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.layers[-6].trainable = True
model.layers[-7].trainable = True
model.layers[-8].trainable = True

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

model.fit(Train_X,Train_y, epochs=15, batch_size=15,validation_data = (test_X, test_y), class_weight = weights)
model.save('EfficientNetB7_Timo.h5')
print("Saved")

del model
del Train_X
del Train_y
del test_X
del test_y
