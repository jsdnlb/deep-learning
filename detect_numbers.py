import numpy as np
from keras import layers, models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt


def main():
    (train_data, train_labels) , (test_data, test_labels) = mnist.load_data()
    """ plt.imshow(train_data[0])
    print(train_labels[0]) """
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='accuracy')
    
    # print(model.summary())
    
    x_train = train_data.reshape((60000, 28*28))
    x_train = x_train.astype('float32')/255
    
    x_test = test_data.reshape((10000, 28*28))
    x_test = x_test.astype('float32')/255
    
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    
    print(train_labels[0])
    print(y_train[0])
    

if __name__ == '__main__':
    main()