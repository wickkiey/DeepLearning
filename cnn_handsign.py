# Convolutional Neural Network on MNIST hand sign data


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (28,28,1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(.4))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(.2))
classifier.add(Dense(units = 25, activation = 'softmax'))

# Compiling the CNN# -*- coding: utf-8 -*-
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#Load mnist hand sign data
import pandas as pd
train = pd.read_csv(r"D:\DataSets\mnisht_hand_sign\sign_mnist_train.csv")
test =  pd.read_csv(r"D:\DataSets\mnisht_hand_sign\sign_mnist_test.csv")
trainX = train.values[:,1:].reshape(train.shape[0],28,28,1).astype("float32")
trainX = trainX/255.0
trainY= train.values[:,0]
testX = test.values[:,1:].reshape(test.shape[0],28,28,1).astype("float32")
testX = testX/255.0
testY = test.values[:,0]

#convert to categorical
import keras
trainY = keras.utils.to_categorical(trainY,25)
testY = keras.utils.to_categorical(testY,25)

#Build a callback mode for graphs
tbcallback = keras.callbacks.TensorBoard(log_dir='D:/Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

classifier.fit(trainX, trainY,
          batch_size=256,
          epochs=7,
          verbose=1,
          validation_data=(testX, testY),callbacks=[tbcallback])

#Test Acc : 95% 
#Validation ACC : 92 %


