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


epoch = 10
BatchSize = 8


# Layer1
# input: all data
# classify: aquatic, caterpillar, 
# segway, helicopter, snowmobile, bike, automobile 
dir1= 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/Layer1'

# Layer Aquatic
# input: class aquatic
# classify: boat barge
dirAq= 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/Aquatic'

# Layer Automobile
# input: automobile
# classify: small vehicles, large vehicles
dirAuto= 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/Automobile'

# Layer Large Vehicles
# input: large vehicles
# Classify: Bus, truck, van
dirLarge= 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/LargeVehicles'

# Layer Small Vehicles
# input: small vehicles
# Classify: cart, taxi, car, limousine
dirSmall= 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/SmallVehicles'

# Layer Van
# input: van
# classify: Van, Ambulance
dirVan= 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/Van'

# Layer Bike
# input: bike
# classify: Bicycle, Motorcycle, Cart
dirBike= 'C:/Users/juspe/Documents/Koodailua/tau-vehicle-37/train/Bike'

#Layer 1

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
'''
#create generators which read images from hard drive only one batch at a time
train_generator = datagen.flow_from_directory(
    directory=dir1,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = ['Aquatic','Automobile','Bike','Caterpillar','Helicopter','Segway','Snowmobile','Tank'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(512, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(8, activation = "sigmoid")(w)
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


#train NN
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=10,
)

model.save('Layer1_Jussi_MobnetV2.h5')
del model
del train_generator


###############
#Layer aquatic#
###############

train_generator = datagen.flow_from_directory(
    directory=dirAq,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = ['Barge','Boat'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(512, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(2, activation = "sigmoid")(w)
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


#train NN
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=10,
)

model.save('LayerAq_Jussi_MobnetV2.h5')
del model
del train_generator

##################
#Layer Automobile#
##################

train_generator = datagen.flow_from_directory(
    directory=dirAuto,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = ['Large','Small'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(512, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(2, activation = "sigmoid")(w)
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


#train NN
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=10,
)

model.save('LayerAuto_Jussi_MobnetV2.h5')
del model
del train_generator

#####################
#Layer LargeVehicles#
#####################

train_generator = datagen.flow_from_directory(
    directory=dirLarge,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = ['Bus','Truck','Van'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(512, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(3, activation = "sigmoid")(w)
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


#train NN
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epoch,
)

model.save('LayerLarge_Jussi_MobnetV2.h5')
del model
del train_generator
'''
#####################
#Layer SmallVehicles#
#####################

train_generator = datagen.flow_from_directory(
    directory=dirSmall,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = ['Car','Cart','Limousine','Taxi'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(512, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(4, activation = "sigmoid")(w)
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


#train NN
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epoch,
)

model.save('LayerSmall_Jussi_MobnetV2.h5')
del model
del train_generator

#####################
#Layer Van#
#####################

train_generator = datagen.flow_from_directory(
    directory=dirVan,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = ['Ambulance','Van'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(512, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(2, activation = "sigmoid")(w)
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


#train NN
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epoch,
)

model.save('LayerVan_Jussi_MobnetV2.h5')
del model
del train_generator

############
#Layer Bike#
############

train_generator = datagen.flow_from_directory(
    directory=dirBike,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=BatchSize,
    classes = ['Bicycle','Cart','Motorcycle'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3),include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(512, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(3, activation = "sigmoid")(w)
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',metrics=['accuracy'])

#get value for batches to generate per epoch
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


#train NN
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epoch,
)

model.save('LayerBike_Jussi_MobnetV2.h5')
del model
del train_generator