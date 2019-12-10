import os 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
import cv2
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
import h5py



directory = "C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/train"

class_names = sorted(os.listdir(directory))


test_files = 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/test/testset'

Test = []


for file in os.listdir(test_files):
        if file.endswith('.jpg'):
            # Load the image:
            img = plt.imread(test_files + os.sep + file)
            # Resize it to the net input size:

            img = cv2.resize(im, (224,224))
            img = img.astype(np.float32)
            img -= 128
            Test.append(img)
            print(file)

Test = np.array(Test)
Pred = np.zeros((Test.shape[0],1))

Layer1 = tf.keras.models.load_model('Layer1_Jussi_MobnetV2.h5')

L1Pred = np.argmax(Layer1.predict(Test),1)
del Layer1

Aquatic = L1Pred[L1Pred.argwhere(0)]
Automobile = L1Pred[L1Pred.argwhere(1)]
Bike = L1Pred[L1Pred.argwhere(2)]
Pred[L1Pred.argwhere(3)] = 7
Pred[L1Pred.argwhere(4)] = 8
Pred[L1Pred.argwhere(5)] = 11
Pred[L1Pred.argwhere(6)] = 12
Pred[L1Pred.argwhere(7)] = 14


LayerAq = tf.keras.models.load_model('LayerAq_Jussi_MobnetV2.h5')

AqPred = np.argmax(LayerAq.predict(Aquatic),1)
AqPred = AqPred.where(AqPred==0,1,3)
Pred[L1Pred.argwhere(0)] = AqPred
del LayerAq

LayerAuto = tf.keras.models.load_model('LayerAuto_Jussi_MobnetV2.h5')

AutoPred = np.argmax(LayerAuto.predict(Automobile),1)
Large = AutoPred[AutoPred.argwhere(0)]
Small = AutoPred[AutoPred.argwhere(1)]
del LayerAuto

LayerLarge = tf.keras.models.load_model('LayerLarge_Jussi_MobnetV2.h5')

LargePred = np.argmax(LayerLarge.predict(Large),1)
LargePred = LargePred.where(LargePred==0,4,LargePred)
LargePred = LargePred.where(LargePred==1,15,LargePred)
Van = LargePred[LargePred.argwhere(2)]
del LayerLarge

LayerVan = tf.keras.models.load_model('LayerVan_Jussi_MobnetV2.h5')

VanPred = np.argmax(LayerVan.predict(Van),1)
VanPred = VanPred.where(VanPred==1,16,VanPred)
LargePred[LargePred.argwhere(2)] = VanPred
AutoPred[AutoPred.argwhere(0)] = LargePred
del LayerLarge

LayerSmall = tf.keras.models.load_model('LayerSmall_Jussi_MobnetV2.h5')

SmallPred = np.argmax(LayerSmall.predict(Small),1)
SmallPred = SmallPred.where(SmallPred==0,5,SmallPred)
SmallPred = SmallPred.where(SmallPred==1,6,SmallPred)
SmallPred = SmallPred.where(SmallPred==2,9,SmallPred)
SmallPred = SmallPred.where(SmallPred==3,14,SmallPred)
AutoPred[AutoPred.argwhere(1)] = SmallPred
Pred[L1Pred.argwhere(1)]= AutoPred
del LayerSmall

LayerBike = tf.keras.models.load_model('LayerBike_Jussi_MobnetV2.h5')

BikePred = np.argmax(LayerBike.predict(Bike),1)
BikePred = BikePred.where(BikePred==0,2,BikePred)
BikePred = BikePred.where(BikePred==1,6,BikePred)
BikePred = BikePred.where(BikePred==2,10,BikePred)
Pred[L1Pred.argwhere(2)] = BikePred
del LayerBike

with open("C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/submissionTree1.csv", "w") as fp:
    fp.write("Id,Category\n")
    i = 0

    for i in range(pred.shape[0]):
        label = class_names[np.argmax(pred[i])] 

        fp.write("%d,%s\n" % (i, label))
        i +=1
