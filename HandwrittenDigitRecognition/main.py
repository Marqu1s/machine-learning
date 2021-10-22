import os
import cv2 #computervision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#dataset from tensorflow
mnist = tf.keras.datasets.mnist

#usually you split your data, 80% training 20% testing
#x is the picture, y is the classification or digit

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# the flatten layer turns the grid into one line of pixels so instead of 
# 28,28 its 784 units 
djdjd