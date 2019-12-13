import os 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model, svm,neighbors, ensemble, discriminant_analysis, model_selection, metrics
import cv2
import tensorflow as tf
import tensorflow.keras

directory = "C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/train"

class_names = sorted(os.listdir(directory))
print(class_names)
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

# Find all image files in the data directory.

X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
'''
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
'''

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
            img = cv2.resize(img, (128,128))
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


# Cast the python lists to a numpy array.
X = np.array(X)
X = X.reshape((X.shape[0],-1))
y = np.array(y)


classifiers = []

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
classifiers.append(lda)

svml = sklearn.svm.SVC(kernel='linear')
classifiers.append(svml)

svmrbf = sklearn.svm.SVC(kernel= 'rbf')
classifiers.append(svmrbf)

lgr = sklearn.linear_model.LogisticRegression()
classifiers.append(lgr)

forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
classifiers.append(forest)

knn = sklearn.neighbors.KNeighborsClassifier()
classifiers.append(knn)

for i in classifiers:
    print(sklearn.model_selection.cross_val_score(i,X,y))
