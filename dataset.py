####################################################
# Neural Networks - UNIVERSITY OF GRONINGEN - 2020 #
####################################################
# Konstantin Rolf - S3750558
# Kacper - (SNumber)
# Nicholas - (SNumber)
# Daniel - (SNumber)

import argparse
import numpy as np
from emnist import extract_training_samples
import random
from PIL import Image
from sklearn.utils import shuffle

"""
Creates a new matrix of letters with the given rows and columns.
The dataset gives a list of numpy images with the mnistSize that
are rearranged into the newly created matrix. Returns
a tuple containing the image matrix as well as the label matrix. 
"""
def createLetterMatrix(dataset, labels, colSize=8, rowSize=8, mnistSize=28):
    word = Image.new('F', (colSize * mnistSize, rowSize * mnistSize))
    newLabels = np.zeros(shape=(rowSize, colSize))

    index = 0
    for y in range(colSize):
        for x in range(rowSize):
            pillowImage = Image.fromarray(dataset[index])
            newLabels[x, y] = labels[index]
            word.paste(pillowImage, (y * mnistSize, x * mnistSize))
            index += 1

    return (np.array(word) / 255.0, newLabels - 1)

"""
Creates a complete dataset of the given length. Specifies the
amount of rows and columns that each image should contains.
"""
def createDataset(length=100, colSize=8, rowSize=8, initialShuffle=True):
    mnistSize = 28
    size = rowSize * colSize
    images, labels = extract_training_samples('letters')
    
    if initialShuffle:
        images, labels = shuffle(images, labels)

    srcIndex = 0
    dataset = np.zeros(shape=(length, rowSize * mnistSize, colSize * mnistSize))
    dataLabels = np.zeros(shape=(length, rowSize, colSize))
    for i in range(length):
        # Reshuffles if the end of the dataset is reached
        if (srcIndex+1) * size >= len(images):
            srcIndex = 0
            images, labels = shuffle(images, labels)
        sourceImages = images[srcIndex * size:(srcIndex+1) * size]
        sourceLables = labels[srcIndex * size:(srcIndex+1) * size]
        srcIndex += 1

        # Creates the image and label matrix
        matrixData, matrixLabels = createLetterMatrix(
            sourceImages, sourceLables,
            rowSize=rowSize, colSize=colSize, mnistSize=mnistSize)
        dataset[i] = matrixData
        dataLabels[i] = matrixLabels

    return (dataset, dataLabels)
    

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Creates a new dataset.')
    
    # Required positional argument
    parser.add_argument('--length', type=int, default=100, help='The amount of images that should be generated.')
    parser.add_argument('--cols', type=int, default=8, help='The amount of columns in each image')
    parser.add_argument('--rows', type=int, default=8, help='The amount of rows in each image')

    args = parser.parse_args()

    print('Creating dataset with length {}, ({}, {})'.format(
        args.length, args.rows, args.cols))
    data, labels = createDataset(
        length=args.length,
        rowSize=args.rows,
        colSize=args.cols
    )
    np.save("data", data)
    np.save("labels", labels)
    print('Dataset successfully created!')
    