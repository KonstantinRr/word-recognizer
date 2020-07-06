####################################################
# Neural Networks - UNIVERSITY OF GRONINGEN - 2020 #
####################################################
# Konstantin Rolf - S3750558
# Kacper - (SNumber)
# Nicholas - (SNumber)
# Daniel - (SNumber)

import argparse
import numpy as np
import cv2
from sklearn.model_selection import *
from dataset import *
import os
import string
import datetime

# Tensorflow GPU config
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

"""
Creates the model that is used to analyze a single digit.
It constructs and compiles a model with the given input shape.
"""
def createModel(input_shape=(28, 28, 1)):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding="same")(input_img)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(26, activation="softmax")(x)

    model = Model(input_img, x)
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["acc"])
    model.summary()
    return model

"""
Takes a source image array and converts it to the correct shape.
The axis is scaled down if it is larger than the target shape.
Otherwise it is filled up with zeros on both sides.
"""
def fillImage(img, shape=(28, 28)):
    # Downscaling
    if img.shape[0] > shape[0]:
        img = cv2.resize(img, (shape[0], img.shape[1]))
    if img.shape[1] > shape[1]:
        img = cv2.resize(img, (img.shape[0], shape[1]))
    
    # Padding
    if img.shape[0] < shape[0]:
        hUneven = img.shape[0] % 2
        hPadding = (shape[0] - img.shape[0]) // 2
        img = np.pad(img, ((hPadding, hPadding + hUneven),
            (0, 0)), mode='constant')
    if img.shape[1] < shape[1]:
        vUneven = img.shape[1] % 2
        vPadding = (shape[1] - img.shape[1]) // 2
        img = np.pad(img, ((0, 0),
            (vPadding, vPadding + vUneven)), mode='constant')
    return img

"""
Splits the source image in different segments containing letters
Segments are split by vertical lines only. A word is considered as
a list of horizontal aligned letters.
Returns a list of images containing the letters. Letters are stored
from left to right in the source image.
"""
def splitSegments(img):
    img = (img * 255).astype("uint8")
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    size_thresh = 1
    intervals = []
    for k in range(1, n_labels):
        if stats[k, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[k, cv2.CC_STAT_LEFT]
            y = stats[k, cv2.CC_STAT_TOP]
            w = stats[k, cv2.CC_STAT_WIDTH]
            h = stats[k, cv2.CC_STAT_HEIGHT]
            
            addInterval = True
            for i in intervals:
                if i[0] <= x <= i[1]:
                    i[1] = max(i[1], x + w)
                    addInterval = False
                if i[0] <= x+w <= i[1]:
                    i[0] = min(i[0], x)
                    addInterval = False

                if x <= i[0] <= x+w:
                    i[1] = max(i[1], x + w)
                    addInterval = False
                if x <= i[1] <= x+w:
                    i[0] = min(i[0], x)
                    addInterval = False

            if addInterval:
                intervals.append([x, x+w])
    intervals = [s for s in intervals if s[1] - s[0] >= 3]
    intervals.sort(key=lambda x: x[0])
    return intervals

"""
Creates and trains the model. It is possible to specify weights
that are loaded in the beginning and a path where the trained
weights will be saved
"""
def train(epochs, loadWeights=None, saveWeights=None):
    xtrain, xtest, ytrain, ytest = createTrainingSet()
    
    model = createModel()
    if loadWeights and loadWeights != '':
        print('Loading model weights {}'.format(loadWeights))
        model.load_weights(loadWeights)

    # Create a callback that saves the model's weights
    checkpoints = []
    if saveWeights:
        checkpoint_dir = os.path.dirname(saveWeights)
        checkpoints.append(ModelCheckpoint(
            filepath=saveWeights,
            save_weights_only=True, verbose=1))

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoints.append(tensorboard_callback)

    model.fit(xtrain, ytrain, batch_size=64,
        epochs=epochs, validation_data=(xtest, ytest),
        callbacks=checkpoints)
    return model
    
"""
Predicts a word inside an image. The function will separate the image
in single letters and predict them with the help of the model.
"""
def predict(loadWeights, img, show=True, boundingBoxes=True):
    # creates the model and loads the weights
    model = createModel()
    model.load_weights(loadWeights)

    # finds the letter segments and creates unified input images
    intervals = splitSegments(img)
    images = np.stack([fillImage(img[0:28, i[0]:i[1]]) for i in intervals])

    # Predicts the images
    result = model.predict(images)
    letters = np.argmax(result, axis=1)
    
    word = ''.join([string.ascii_lowercase[i] for i in letters])
    if show:
        print('RESULT:', word)
        if boundingBoxes:
            for i in intervals:
                cv2.rectangle(img, (i[0], -1), (i[1], 28), 255, thickness=1)
        cv2.namedWindow("Train", cv2.WINDOW_NORMAL)
        cv2.imshow("Train", img)
        cv2.waitKey(0)
    return word

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Creates a new dataset.')
    
    # Required positional argument
    parser.add_argument('type', type=str, help='Either train or predict.')
    parser.add_argument('--weights', type=str, help='If the program should load some weights.')
    parser.add_argument('--save', type=str, default='training/check', help='If the program should load some weights.')
    parser.add_argument('--epochs', type=int, default=10, help='The amount of epochs that the program should train.')
    parser.add_argument('--source', type=str, default="source.png", help='The amount of rows in each image')

    args = parser.parse_args()
    if args.type == 'train':
        train(args.epochs, loadWeights=args.weights, saveWeights=args.save)
    elif args.type == 'predict':
        if args.weights is None:
            print("You need to specify --weights when predicting images.")
            exit(0)

        data, labels = createDataset(length=1, rowSize=1, colSize=8)
        word = predict(args.weights, data[0])
        print(word)
    
    #train()