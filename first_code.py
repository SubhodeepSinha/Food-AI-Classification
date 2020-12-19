
#tensorflow
import tensorflow as tf
from tensorflow import keras

#helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout

classifier = Sequential()

#1st convolution
classifier.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(3, 3)))

#2nd convolution
classifier.add(Conv2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(3, 3)))

#flattening
classifier.add(Flatten())


classifier.add(Dense(units=64,activation = 'relu'))

classifier.add(Dense(units=128,activation = 'relu'))

classifier.add(Dense(units=10,activation = 'softmax'))

#Compiling CNN
classifier.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

classifier.summary()





