####################################################
# Neural Networks - UNIVERSITY OF GRONINGEN - 2020 #
####################################################
# Konstantin Rolf - S3750558
# Kacper - (SNumber)
# Nicholas - (SNumber)
# Daniel - (SNumber)


import numpy as np
import cv2
from sklearn.model_selection import *
from dataset import *

# Tensorflow GPU config
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

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
Returns a list of split images
"""
def splitSegments(img):
    img = (img * 255).astype("uint8")
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    size_thresh = 1
    intervals = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
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
    for i in intervals:
        cv2.rectangle(img, (i[0], -1), (i[1], 28), 255, thickness=1)

    images = [fillImage(img[0:28, i[0]:i[1]]) for i in intervals]    
    return images


def train():
    data, labels = extract_training_samples('letters')
    labels = labels - 1
    
    labels = to_categorical(labels)
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.25)
    xtrain = np.expand_dims(xtrain, -1)
    xtest = np.expand_dims(xtest, -1)

    print(labels.shape)
    
    model = createModel()
    model.fit(xtrain, ytrain, batch_size=64, epochs=10, validation_data=(xtest, ytest))
    
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Test", data[0])
    cv2.waitKey(0)
    

if __name__ == '__main__':
    data, labels = createDataset(length=1, rowSize=1, colSize=8)
    #cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    #cv2.imshow("Test", data[0])
    #cv2.waitKey(0)
    splitSegments(data[0])
    #train()