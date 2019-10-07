#
#Summer Project--Particle Shape Analysis
#
#
# import the necessary packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.utils import to_categorical
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import adam
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import random
import cv2
import os
import pandas as pd
from imutils import paths
from imutils import build_montages
import imutils


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(10))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model


# initialize the number of epochs to train for, initia learning rate,
# and batch size
NUM_EPOCHS=50
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('/Users/poyen/Desktop/sand/train')))
random.seed(42)
random.shuffle(imagePaths)


# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
print("[INFO] converting image to array...")
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image1 = cv2.imread(imagePath)
    image1 = cv2.resize(image1, (28, 28))
    image1 = img_to_array(image1)
    data.append(image1)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    if label == "0.1":
        label = 0
    elif label == "0.2":
        label = 1
    elif label == "0.3":
        label = 2
    elif label == "0.4":
        label = 3
    elif label == "0.5":
        label = 4
    elif label == "0.6":
        label = 5
    elif label == "0.7":
        label = 6
    elif label == "0.8":
        label = 7
    elif label == "0.9":
        label = 8
    elif label == "1":
        label = 9
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=10)

opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="mean_squared_error", optimizer=opt,
	metrics=["accuracy"])
model.summary()
# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=NUM_EPOCHS, verbose=2)

score = model.evaluate(testX, testY, batch_size=BS)
print(score)

# save the model to disk
print("[INFO] serializing network...")
model.save('my_model.h5')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = NUM_EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Particle Sharpness")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="right")
plt.savefig("loss")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = NUM_EPOCHS
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Accuracy on Particle Sharpness")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="right")
plt.savefig("Acc")

# load the image
results=[]
imagePaths1 = list(paths.list_images('/Users/poyen/Desktop/sand/test/image'))
for imagePath in imagePaths1:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)

    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('my_model.h5')

    # classify the input image
    pred=model.predict(image)[0]
    pred=np.argmax(pred)

    # build the label
    if pred == 0:
        label = 0.1
    elif pred == 1:
        label = 0.2
    elif pred == 2:
        label = 0.3
    elif pred == 3:
        label = 0.4
    elif pred == 4:
        label = 0.5
    elif pred == 5:
        label = 0.6
    elif pred == 6:
        label = 0.7
    elif pred == 7:
        label = 0.8
    elif pred == 8:
        label = 0.9
    elif pred == 9:
        label = 1

    print("The sharpness of{} is {}".format(imagePath,label))

    color = (0, 0, 255) if pred <= 3 else (0, 255, 0)
 
    # resize our original input (so we can better visualize it) and
    # then draw the label on the image
    orig = cv2.resize(orig, (64, 64))
    cv2.putText(orig, str(label), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 2)

    # add the output image to our list of results
    results.append(orig)
# create a montage using 64x64 "tiles" with 6 rows and 6 columns
montage = build_montages(results, (64, 64), (6, 6))[0]
 
# show the output montage
cv2.imshow("Results", montage)
cv2.waitKey(0)
