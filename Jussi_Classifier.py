
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

model1 = tf.keras.models.load_model('ResNet50_Jussi.h5')
#model2 = tf.keras.models.load_model('Mobilenet_Jussi.h5')
model3 = tf.keras.models.load_model('MobilenetV2_Jussi.h5')
model4 = tf.keras.models.load_model('InceptionV3_Jussi.h5')

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
#bias = np.zeros((128,128,1))

data = []

for root, dirs, files in os.walk(directory):
    #print(dirs)
    for name in files:
        #print(files)
        if name.endswith('.jpg'):
            path = os.path.join(root, name)
            print(path)
            # Load the image:
            im = plt.imread(root + os.sep + name)
            # Resize it to the net input size:

            img = cv2.resize(im, (128,128))
            img = img.astype(np.float32)
            img -= 128
            pred1 = model1.predict(img[np.newaxis, ...])
            #pred2 = model2.predict(img[np.newaxis, ...])
            pred3 = model3.predict(img[np.newaxis, ...])
            pred4 = model4.predict(img[np.newaxis, ...])
            pred = [np.argmax(pred1),np.argmax(pred3),np.argmax(pred4)]
            data.append(pred)


            img2 = cv2.resize(im, (224,224))
            img2 = img2.astype(np.float32)
            img2 -= 128
            x = model.predict(img2[np.newaxis, ...])[0]
            # Convert the data to float, and remove mean          
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
data = np.array(data)


svmrbf = sklearn.svm.SVC(kernel= 'rbf')
svmrbf.fit(X,y)

pred = np.reshape(svmrbf.predict(X),(np.size(y),1))
np.concatenate((data,pred),1)

#############################
# Test if idea works at all #                        
#############################
Train_X, test_X, Train_y, test_y= sklearn.model_selection.train_test_split(data,y,test_size=0.2)

test_gnb = sklearn.naive_bayes.GaussianNB()
test_gnb.fit(Train_X,Train_y)
print(sklearn.metrics.accuracy_score(test_y,test_gnb.predict(test_X)))


# reclaim memory
del test_gnb
del Train_X
del test_X
del Train_y
del test_y
#####################################################
# Train w full dataset

gnb = sklearn.naive_bayes.GaussianNB()
gnb.fit(data,y)

####################################################

test_files = 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/test/testset'

Test = []

for file in os.listdir(test_files):
        if file.endswith('.jpg'):
            # Load the image:
            im = plt.imread(test_files + os.sep + file)
            # Resize it to the net input size:

            img = cv2.resize(im, (128,128))
            img = img.astype(np.float32)
            img -= 128
            pred1 = model1.predict(img[np.newaxis, ...])
            #pred2 = model2.predict(img[np.newaxis, ...])
            pred3 = model3.predict(img[np.newaxis, ...])
            pred4 = model4.predict(img[np.newaxis, ...])
            
            
            img2 = cv2.resize(im, (224,224))
            img2 = img2.astype(np.float32)
            img2 -= 128
            x = model.predict(img2[np.newaxis, ...])[0]
            linearpred = svmrbf.predict(X)
            pred = [np.argmax(pred1),np.argmax(pred3),np.argmax(pred4),linearpred]
            Test.append(pred)

Test = np.array(Test)            

with open("C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/submission.csv", "w") as fp:
    fp.write("Id,Category\n")
    i = 0

    pred = gnb.predict(Test)
    for prediction in pred:
        label = class_names[prediction] 

        fp.write("%d,%s\n" % (i, label))
        i +=1


