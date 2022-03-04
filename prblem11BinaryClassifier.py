from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.utils import to_categorical


def prepare_data():
    # load image data
    (trainX, trainY), (testX, testY) = mnist.load_data()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    index = np.argwhere(trainY < 2)
    trainX = trainX[index]
    trainY = trainY[index]
    trainX = trainX[:, 0, :, :]
    trainY = trainY[:, 0]

    index = np.argwhere(testY < 2)
    testX = testX[index]
    testY = testY[index]
    testX = testX[:, 0, :, :]
    testY = testY[:, 0]

    classN = 2
    trainY = to_categorical(trainY, classN)
    testY = to_categorical(testY, classN)

    # to convert pixel values from 0-255 to 0-1.

    trainX = trainX.astype(np.float32)
    testX = testX.astype(np.float32)
    trainX /= 255
    testX /= 255
    return trainX, trainY, testX, testY


def build_model():
    inputs = Input((28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(inputs)
    x = MaxPool2D(2, 2)(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = MaxPool2D(2, 2)(x)
    x = Flatten()(x)
    outputs = Dense(2, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.summary()
    return model


def main():
    trainX, trainY, testX, testY = prepare_data()

    model = build_model()
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(trainX, trainY, epochs= 12, validation_split=.2)
    model.compile(metrics= 'accuracy')
    model.evaluate(testX, testY)


if __name__ == '__main__':
    main()
