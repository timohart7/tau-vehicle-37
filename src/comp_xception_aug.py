# author: jussikai, timoh

import os 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Model

trainDir = "data/train/train"
class_names = sorted(os.listdir(trainDir))

weights = [4,3,1,1,1,1,5,3,2,5,1,4,4,3,2,1,1]

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 45, height_shift_range= 80, 
    horizontal_flip = True, vertical_flip = False, width_shift_range=80, zoom_range=[0.5,1.0])

train_generator = train_datagen.flow_from_directory(
    directory=trainDir,
    target_size=(224,224),
    color_mode="rgb",
    batch_size=16,
    classes=class_names,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

################
# Xception #
################

base_model = base_model = tf.keras.applications.xception.Xception(weights='imagenet', input_shape = (224,224,3), include_top = False)
#base_model.summary()

w = base_model.output
w = Flatten()(w)
w = Dense(256, activation = "relu")(w)
w = Dense(128, activation = "relu")(w)
output = Dense(17, activation = "sigmoid")(w)

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model = Model(inputs = [base_model.inputs[0]], outputs = [output])

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=16,
                    class_weight=weights
)

model.save('Xception_Timo_224x224ep16aug.h5')
print("Saved")
