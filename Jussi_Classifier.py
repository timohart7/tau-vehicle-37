
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
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import h5py
import efficientnet.tfkeras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


directory = "C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/train"

class_names = sorted(os.listdir(directory))


def majority_vote(prediction):
    y = np.sum(np.array(prediction),1)

    return np.argmax(y)



test_files = 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/test/testset'

TestNN128 = []
TestNN224 =[]


for file in os.listdir(test_files):
        if file.endswith('.jpg'):
            # Load the image:
            im = plt.imread(test_files + os.sep + file)
            # Resize it to the net input size:

            img = cv2.resize(im, (128,128))
            img = img.astype(np.float32)
            img -= 128
            TestNN128.append(img)

            img2 = cv2.resize(im, (224,224))
            img2 = img2.astype(np.float32)
            img2 -= 128
            TestNN224.append(img2)
            print(file)

TestNN128 = np.array(TestNN128)   
TestNN224 = np.array(TestNN224)         

#128 models
model1 = tf.keras.models.load_model('ResNet50_Jussi.h5')
model2 = tf.keras.models.load_model('InceptionV3_Jussi_v2.h5')

#224 models
model3 = tf.keras.models.load_model('InceptionV3_v4.h5')
model4 = tf.keras.models.load_model('MobilenetV2_Jussi_v3.h5')
model5 = tf.keras.models.load_model('ResNet50_v2.h5')

pred0 = np.array(model1.predict(TestNN128)).reshape((-1,17,1))
pred1 = np.array(model2.predict(TestNN128)).reshape((-1,17,1))
del TestNN128
pred2 = np.array(model3.predict(TestNN224)).reshape((-1,17,1))
pred3 = np.array(model4.predict(TestNN224)).reshape((-1,17,1))
pred4 = np.array(model5.predict(TestNN224)).reshape((-1,17,1))



pred = np.concatenate((pred0,pred1,pred2,pred3,pred4), axis = 2)


with open("C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/submission10.csv", "w") as fp:
    fp.write("Id,Category\n")
    i = 0



    for i in range(pred.shape[0]):
        label = class_names[majority_vote(pred[i,:,:])] 

        fp.write("%d,%s\n" % (i, label))
        i +=1


