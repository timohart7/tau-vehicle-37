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


'''
directory = "C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/train"

class_names = sorted(os.listdir(directory))


test_files = 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/test/testset'

Test = []


for file in os.listdir(test_files):
        if file.endswith('.jpg'):
            # Load the image:
            img = plt.imread(test_files + os.sep + file)
            # Resize it to the net input size:

            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32)
            img -= 128
            Test.append(img)
            print(file)

Test = np.array(Test)
'''
def JussiTree(Test):

    Pred = np.zeros((Test.shape[0]),dtype=int)

    Layer1 = tf.keras.models.load_model('Layer1_Jussi_MobnetV2.h5')

    L1Pred = np.argmax(Layer1.predict(Test),1)
    del Layer1

    Aquatic = Test[np.argwhere(L1Pred==0).ravel()]
    Automobile = Test[np.argwhere(L1Pred==1).ravel()]
    Bike = Test[np.argwhere(L1Pred==2).ravel()]
    Pred[np.argwhere(L1Pred==3)] = 7
    Pred[np.argwhere(L1Pred==4)] = 8
    Pred[np.argwhere(L1Pred==5)] = 11
    Pred[np.argwhere(L1Pred==6)] = 12
    Pred[np.argwhere(L1Pred==7)] = 13
    del Test
    tf.keras.backend.clear_session()

    LayerAq = tf.keras.models.load_model('LayerAq_Jussi_MobnetV2.h5')

    AqPred = np.argmax(LayerAq.predict(Aquatic),1)
    AqPred = np.where(AqPred==0,1,3)
    Pred[np.argwhere(L1Pred==0).ravel()] = AqPred
    del LayerAq
    del Aquatic
    tf.keras.backend.clear_session()

    LayerBike = tf.keras.models.load_model('LayerBike_Jussi_MobnetV2.h5')

    BikePred = LayerBike.predict(Bike)
    BikePred = np.argmax(BikePred,1)
    BikePred = np.where(BikePred==2,10,BikePred)
    BikePred = np.where(BikePred==0,2,BikePred)
    BikePred = np.where(BikePred==1,6,BikePred)
    Pred[np.argwhere(L1Pred==2).ravel()] = BikePred
    del LayerBike
    del Bike
    tf.keras.backend.clear_session()

    LayerAuto = tf.keras.models.load_model('LayerAuto_Jussi_MobnetV2.h5')

    AutoPred = np.argmax(LayerAuto.predict(Automobile),1)
    Large = Automobile[np.argwhere(AutoPred==0).ravel()]
    Small = Automobile[np.argwhere(AutoPred==1).ravel()]
    del LayerAuto
    del Automobile
    tf.keras.backend.clear_session()

    LayerLarge = tf.keras.models.load_model('LayerLarge_Jussi_MobnetV2.h5')

    LargePred = np.argmax(LayerLarge.predict(Large),1)
    LargePred = np.where(LargePred==0,4,LargePred)
    LargePred = np.where(LargePred==1,15,LargePred)
    Van = Large[np.argwhere(LargePred==2).ravel()]
    del LayerLarge
    del Large
    tf.keras.backend.clear_session()

    LayerVan = tf.keras.models.load_model('LayerVan_Jussi_MobnetV2.h5')

    VanPred = np.argmax(LayerVan.predict(Van),1)
    VanPred = np.where(VanPred==1,16,VanPred)
    LargePred[np.argwhere(LargePred==2).ravel()] = VanPred
    AutoPred[np.argwhere(AutoPred==0).ravel()] = LargePred
    del LayerVan
    del Van
    del LargePred
    del VanPred
    tf.keras.backend.clear_session()

    LayerSmall = tf.keras.models.load_model('LayerSmall_Jussi_MobnetV2.h5')

    SmallPred = np.argmax(LayerSmall.predict(Small),1)
    SmallPred = np.where(SmallPred==0,5,SmallPred)
    SmallPred = np.where(SmallPred==1,6,SmallPred)
    SmallPred = np.where(SmallPred==2,9,SmallPred)
    SmallPred = np.where(SmallPred==3,14,SmallPred)
    AutoPred[np.argwhere(AutoPred==1).ravel()] = SmallPred
    Pred[np.argwhere(L1Pred==1).ravel()]= AutoPred
    del LayerSmall
    del Small
    del AutoPred
    del SmallPred
    tf.keras.backend.clear_session()

    return(Pred)

'''
with open("C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/submissionTree1.csv", "w") as fp:
    fp.write("Id,Category\n")
    i = 0

    for i in range(Pred.shape[0]):
        label = class_names[Pred[i]] 

        fp.write("%d,%s\n" % (i, label))
        i +=1
'''