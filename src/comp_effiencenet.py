# author: jussikai

import os 
import keras
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras



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


# Find all image files in the data directory.

X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
#bias = np.zeros((128,128,1))

################
# DATA INPUT 1 #
# WHOLE DATA SET, No Agmentation#
################

#for root, dirs, files in os.walk(directory):
#    #print(dirs)
#    for name in files:
#        #print(files)
#        if name.endswith('.jpg'):
#            path = os.path.join(root, name)
#            print(path)
#            # Load the image:
#            img = plt.imread(root + os.sep + name)
#            # Resize it to the net input size:
#            img = cv2.resize(img, (224,224))
#            img = img.astype(np.float32)
#            img -= 128
#            #img = np.concatenate((img,bias),axis=2)
#
#            # Convert the data to float, and remove mean          
#            # And append the feature vector to our list.
#            X.append(img)
#            print(name)
#            print(os.sep)
#            # Extract class name from the directory name:
#            label = path.split(os.sep)[-2]
#            print(label)
#            y.append(class_names.index(label))
#            print(name)
#
#X = np.array(X)
#y = tf.keras.utils.to_categorical(np.array(y))
#
#
#Train_X, test_X, Train_y, test_y= sklearn.model_selection.train_test_split(X,y,test_size=0.2)
#del X
#del y

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

################
# MobileNet_v2 #
# For Model Initiliazation #
################

#base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
##base_model.summary()
#
#w = base_model.output
#w = Flatten()(w)
#w = Dense(256, activation = "relu")(w)
#w = Dense(128, activation = "relu")(w)
#output = Dense(17, activation = "sigmoid")(w)
## Compile the model for execution. Losses and optimizers
## can be anything here, since we don’t train the model.
#model = Model(inputs = [base_model.inputs[0]], outputs = [output])
#
#model.layers[-6].trainable = True
#model.layers[-7].trainable = True
#model.layers[-8].trainable = True
#
#model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])
#
#model.fit(Train_X,Train_y, epochs=15, batch_size=15,validation_data = (test_X, test_y), class_weight = weights)
#model.save('MobilenetV2_Jussi_v2.h5')
#
#del model

################
# comp_Model re_train #
# For Model additional training#
################

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

#from keras_efficientnets import EfficientNetB6
import efficientnet.tfkeras
base_model = efficientnet.tfkeras.EfficientNetB4(input_shape=(224,224,3), classes=17, include_top=False, weights='imagenet')

w = base_model.output
w = Flatten()(w)
w = Dense(256, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(17, activation = "sigmoid")(w)

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.layers[-6].trainable = True
model.layers[-7].trainable = True
model.layers[-8].trainable = True


#train NN
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EpochSize,
                    class_weight= weights
)




model.save('EfficientNetB5_Comp.h5')





