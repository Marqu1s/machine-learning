import os
import cv2 #computervision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#dataset from tensorflow
#mnist = tf.keras.datasets.mnist

#usually you split your data, 80% training 20% testing
#x is the picture, y is the classification or digit

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# the flatten layer turns the grid into one line of pixels so instead of 
# 28,28 its 784 units 
#add dense layer
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))


#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=3)

#model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1

#loss, accuracy = model.evaluate(x_test,y_test)

#print(loss)
#print(accuracy)