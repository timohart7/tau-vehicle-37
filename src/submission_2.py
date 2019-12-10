import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
directory = "C:/Data/tau-vehicle-37/tau-vehicle-37/data/train/train"

class_names = sorted(os.listdir(directory))

print(class_names)


test_files = 'C:/Data/tau-vehicle-37/tau-vehicle-37/data/test/testset'

Test = []

for file in os.listdir(test_files):
        if file.endswith('.jpg'):
            # Load the image:
            img = plt.imread(test_files + os.sep + file)
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32)
            img -= 128
            #img = np.concatenate((img,bias),axis=2)

            # Convert the data to float, and remove mean          
            # And append the feature vector to our list.
            Test.append(img)
            print(file)

Test = np.array(Test)

model = tf.keras.models.load_model('C:/Data/tau-vehicle-37/tau-vehicle-37/src/EfficientNetB4_Comp_re3.h5')

pred=model.predict(Test)


with open("C:/Data/tau-vehicle-37/tau-vehicle-37/subs/submissionMikko2.csv", "w") as fp:
    fp.write("Id,Category\n")
    i = 0

    #pred = model.predict(Test)
    for i in range(pred.shape[0]):
        label = class_names[np.argmax(pred[i])] 

        fp.write("%d,%s\n" % (i, label))
        i +=1
