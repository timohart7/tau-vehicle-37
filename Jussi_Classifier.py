
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
import sklearn
from sklearn import linear_model, svm, ensemble, discriminant_analysis, model_selection, metrics, naive_bayes
import h5py

def majority_vote(prediction):
    y = np.sum(tf.keras.utils.to_categorical(np.array(prediction),17),0)

    if np.max(y) ==1:
        return prediction[0]
    return np.argmax(y)





base_model = tf.keras.applications.mobilenet.MobileNet(input_shape = (224,224,3),include_top = False)
base_model.summary

in_tensor = base_model.inputs[0]# Grab the input of base model
out_tensor = base_model.outputs[0]# Grab the output of base model

# Add an average pooling layer (averaging each of the 1024 channels):
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs  = [in_tensor],outputs = [out_tensor])

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')


directory = "C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/train"

class_names = sorted(os.listdir(directory))

print(class_names)


# Find all image files in the data directory.

X = [] # Feature vectors will go here.
y = [] # Class ids will go here.


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
            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128
            # Push the data through the model:
            x = model.predict(img[np.newaxis, ...])[0]
            # And append the feature vector to our list.
            X.append(x)
            print(name)
            print(os.sep)
            # Extract class name from the directory name:
            label = path.split(os.sep)[-2]
            print(label)
            y.append(class_names.index(label))
            print(name)

X = np.array(X)
y = np.array(y)


svmrbf = sklearn.svm.SVC(kernel= 'rbf')
svmrbf.fit(X,y)


del X
del y
test_files = 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/test/testset'

TestNN = []
TestSVM =[]


for file in os.listdir(test_files):
        if file.endswith('.jpg'):
            # Load the image:
            im = plt.imread(test_files + os.sep + file)
            # Resize it to the net input size:

            img = cv2.resize(im, (128,128))
            img = img.astype(np.float32)
            img -= 128
            TestNN.append(img)

            img2 = cv2.resize(im, (224,224))
            img2 = img2.astype(np.float32)
            img2 -= 128
            x = model.predict(img2[np.newaxis, ...])[0]
            TestSVM.append(x)
            print(file)

TestNN = np.array(TestNN)   
TestSVM = np.array(TestSVM)         

model1 = tf.keras.models.load_model('ResNet50_Jussi.h5')
model3 = tf.keras.models.load_model('MobilenetV2_Jussi.h5')
model4 = tf.keras.models.load_model('InceptionV3_Jussi_v2.h5')

pred0 = np.argmax(np.array(model1.predict(TestNN)),1).reshape((-1,1))
pred1 = np.argmax(np.array(model4.predict(TestNN)),1).reshape((-1,1))
pred2 = np.argmax(np.array(model3.predict(TestNN)),1).reshape((-1,1))
pred3 = np.array(svmrbf.predict(TestSVM)).reshape((-1,1))

pred = np.concatenate((pred0,pred1,pred2,pred3), axis = 1)


with open("C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/submission3.csv", "w") as fp:
    fp.write("Id,Category\n")
    i = 0



    for i in range(pred.shape[0]):
        label = class_names[majority_vote(pred[i])] 

        fp.write("%d,%s\n" % (i, label))
        i +=1


