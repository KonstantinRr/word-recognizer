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
are randomly rearranged into the newly created matrix. Returns
a tuple containing the image matrix and the label matrix. 
"""
def createLetterMatrix(dataset, labels, rowSize=8, colSize=8, mnistSize=28):
    word = Image.new('F', (rowSize * mnistSize, colSize * mnistSize))
    newLabels = np.zeros(shape=(colSize, rowSize))
    shuffledData, shuffledLabels = shuffle(dataset, labels)

    index = 0
    for x in range(rowSize):
        for y in range(colSize):
            pillowImage = Image.fromarray(shuffledData[index])
            newLabels[y, x] = shuffledLabels[index]
            word.paste(pillowImage, (x * mnistSize, y * mnistSize))
            index += 1

    return (np.array(word) / 255.0, newLabels - 1)

"""
Creates a complete dataset of the given length. Specifies the
amount of rows and columns that each image should contains.
"""
def createDataset(length=100, rowSize=8, colSize=8):
    mnistSize = 28
    images, labels = extract_training_samples('letters')

    dataset = np.zeros(shape=(length, colSize * 28, rowSize * 28))
    dataLabels = np.zeros(shape=(length, colSize, rowSize))
    for i in range(length):
        matrix, matrixLabels = createLetterMatrix(images, labels,
            rowSize=rowSize, colSize=colSize, mnistSize=mnistSize)
        dataset[i] = matrix
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
    