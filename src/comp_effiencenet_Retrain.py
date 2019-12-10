# author: jussikai

import os 
import keras
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras
from keras.layers import Activation



########
#Set UP#
########

###########################################################################
#set file paths
Train_directory = "C:/Users/Mikko/Documents/tau-vehicle-37/data/train/train"
Val_directory = Val_directory = "C:/Users/Mikko/Documents/tau-vehicle-37/data/Validation"

#set Batch size (int) set as high as you can without gettim OOM error
BatchSize = 8
EpochSize = 10

#Set Weight according to data amounts in each class
weights = [5,4,1,1,1,1,5,2,2,5,1,4,5,3,2,1,1]
############################################################################

PersonalDirectory = "C:/Users/Mikko/Documents/"
#PersonalDirectory = "C:/Users/juspe/Documents/Koodailua"

directory = "C:/Users/Mikko/Documents/tau-vehicle-37/data/train/train"
class_names = sorted(os.listdir(directory))

print(class_names)



################
# DATA INPUT 2 #
# PATCH SIZE ONLY, WIHT AUGMENTED DATA#
################

#alternative way of getting images
#create a image augmentor
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 45, height_shift_range= 80, 
horizontal_flip = False, vertical_flip = True, width_shift_range=80, zoom_range=[0.5,1.0])

#No Augmentation for val data
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

#create generators which read images from hard drive only one batch at a time
train_generator = train_datagen.flow_from_directory(
    directory=Train_directory,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = class_names,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    directory=Val_directory,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = class_names,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

#Import trained model:
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
model = tf.keras.models.load_model('EfficientNetB4_Comp_re2.h5')

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EpochSize,
                    class_weight= weights
)


model.save('EfficientNetB4_Comp_re3.h5')





