####################################################
# Neural Networks - UNIVERSITY OF GRONINGEN - 2020 #
####################################################
# Konstantin Rolf - S3750558
# Kacper - (SNumber)
# Nicholas - (SNumber)
# Daniel - (SNumber)


import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
import cv2
from sklearn.model_selection import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from dataset import *

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

def fillImage(img, shape=(28, 28)):
    # TODO automatic scaling when larger than 28
    uneven = img.shape[1] % 2
    p = (28 - img.shape[1]) // 2
    img = np.pad(img, ((0, 0), (p, p + uneven)), mode='constant')
    return img

def splitSegments(img, count=8):
    img = (img * 255).astype("uint8")
    gray = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
    #for i in intervals:
    #    cv2.rectangle(img, (i[0], -1), (i[1], 28), 255, thickness=1)

    images = [fillImage(img[0:28, i[0]:i[1]]) for i in intervals]
    """
    print(intervals)
    print(img.shape)
    print(images[0].shape)
    # image, contours, hier
    filled = fillImage(images[0])
    print(filled.shape)
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Test", filled)
    cv2.waitKey(0)
    #vertLines = np.zeros(data.shape[])
    """
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
    #data, labels = createDataset(length=1000, rowSize=1, colSize=1)
    #splitSegments(data[0])
    train()