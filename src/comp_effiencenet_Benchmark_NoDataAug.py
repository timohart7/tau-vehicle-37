# author: jussikai

import os 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
import cv2
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Model
PersonalDirectory = "C:/Users/Mikko/Documents/"
#PersonalDirectory = "C:/Users/juspe/Documents/Koodailua"

directory = "C:/Users/Mikko/Documents/tau-vehicle-37/data/train/train"
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

weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#weights = [4,3,1,1,1,1,4,2,2,4,1,3,4,3,2,1,1]
Train_X, test_X, Train_y, test_y= sklearn.model_selection.train_test_split(X,y,test_size=0.2)
del X
del y

#Import trained model:
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
model = tf.keras.models.load_model('EfficientNetB5_Comp.h5')

model.fit(Train_X,Train_y, epochs=10, batch_size=8,validation_data = (test_X, test_y), class_weight = weights)
model.save('Eff_comp_noDataAug_benchmark.h5')

del model

del Train_X
del Train_y
del test_X
del test_y

