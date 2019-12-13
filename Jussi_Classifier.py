
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
from NNTreeJussi_Test import JussiTree

directory = "train/train"

class_names = sorted(os.listdir(directory))


def majority_vote(prediction):
    #y = np.sum(tf.keras.utils.to_categorical(np.array(prediction),17),0)
    y = np.sum(prediction,1)
    #if np.max(y) ==2:
    #    return prediction[0]
    return np.argmax(y)



test_files = 'test/testset'

#TestNN128 = []
TestNN224 =[]


for file in os.listdir(test_files):
        if file.endswith('.jpg'):
            # Load the image:
            im = plt.imread(test_files + os.sep + file)
            # Resize it to the net input size:

            #img = cv2.resize(im, (128,128))
            #img = img.astype(np.float32)
            #img -= 128
            #TestNN128.append(img)

            img2 = cv2.resize(im, (224,224))
            img2 = img2.astype(np.float32)
            img2 -= 128
            TestNN224.append(img2)
            print(file)

#TestNN128 = np.array(TestNN128)   
TestNN224 = np.array(TestNN224)         
'''
#128 models
model1 = tf.keras.models.load_model('ResNet50_Jussi.h5')
model2 = tf.keras.models.load_model('InceptionV3_Jussi_v2.h5')

pred0 = np.argmax(np.array(model1.predict(TestNN128)),1).reshape((-1,1))
pred1 = np.argmax(np.array(model2.predict(TestNN128)),1).reshape((-1,1))
del TestNN128
tf.keras.backend.clear_session()
'''


#224 models
model0 = tf.keras.models.load_model('Eff_comp_noDataAug_re2.h5')
pred0 = np.array(model0.predict(TestNN224)).reshape((-1,17,1))
del model0
tf.keras.backend.clear_session()
model1 = tf.keras.models.load_model('EfficientNetB7_Timo_224x224ep15.h5')
pred1 = np.array(model1.predict(TestNN224)).reshape((-1,17,1))
del model1
tf.keras.backend.clear_session()
#pred6 = JussiTree(TestNN224).reshape((-1,1))

model4 = tf.keras.models.load_model('EffNetB4.h5')
pred4 = np.array(model4.predict(TestNN224)).reshape((-1,17,1))
del model4
tf.keras.backend.clear_session()

model2 = tf.keras.models.load_model('InceptionV3_v4.h5')
model3 = tf.keras.models.load_model('MobilenetV2_Jussi_v3.h5')
#model5 = tf.keras.models.load_model('ResNet50_v2.h5')


pred2 = np.array(model2.predict(TestNN224)).reshape((-1,17,1))
pred3 = np.array(model3.predict(TestNN224)).reshape((-1,17,1))
#pred4 = np.argmax(np.array(model5.predict(TestNN224)),1).reshape((-1,1))
del model2
del model3
tf.keras.backend.clear_session()

model5 = tf.keras.models.load_model('Xception_Timo_224x224ep16aug.h5')
pred5 = np.array(model5.predict(TestNN224)).reshape((-1,17,1))
del model5
tf.keras.backend.clear_session()



pred = np.concatenate((pred0,pred0,pred1,pred1,pred2,pred3,pred4,pred4,pred5), axis = 2)


with open("submissionMajvote_13_12_3.csv", "w") as fp:
    fp.write("Id,Category\n")
    #fp.write("Id,128ResNet,128Inception,Inception,MobNetV2,ResNet50,Tree\n")
    i = 0



    for i in range(pred.shape[0]):
        label = class_names[majority_vote(pred[i,:,:])] 

        fp.write("%d,%s\n" % (i, label))
        #fp.write("%d,%s,%s,%s,%s,%s,%s\n" % (i, pred0[i],pred1[i],pred2[i],pred3[i],pred4[i],pred6[i]))
        i +=1


