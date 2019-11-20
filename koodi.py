import sklearn
import tensorflow as tf
import tf.keras

class_names =sorted(os.listdir(r"C:\Work\sgndataset\train\"))

base_model = tf.keras.applications.mobilenet.MobileNet(input_shape = (224,224,3),include_top = False)

in_tensor = base_model.inputs[0]# Grab the input of base model
out_tensor = base_model.outputs[0]# Grab the output of base model

# Add an average pooling layer (averaging each of the 1024 channels):
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs  = [in_tensor],outputs = [out_tensor])# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss = "categorical_crossentropy", optimizer = ’sgd’)

# Find all image files in the data directory.

X = []# Feature vectors will go here.
y = []# Class ids will go here.

for root, dirs, filesinos.walk(r"C:\Work\sgndataset\train\"):
    for name in files:
        if name.endswith(".jpg"):
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

            # Extract class name from the directory name:
            label = name.split(os.sep)[-2]
            y.append(class_names.index(label))

# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)