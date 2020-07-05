# Neural Networks - UNIVERSITY OF GRONINGEN - 2020
# Konstantin Rolf - S3750558
# Kacper - (SNumber)
# Nicholas Koundouros - (SNumber)
# Daniel - (SNumber)

import tensorflow as tf
import numpy as np
from emnist import extract_training_samples
import cv2
from PIL import Image

if __name__ == '__main__':
    print('Runnning Word recognizer')
    images, labels = extract_training_samples('digits')
    print(images.shape)