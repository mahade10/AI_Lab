from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model


def build_model():
    inputs = Input((28, 28, 1))
    x = Conv2D(filters=3, kernel_size=(2, 2), strides=(2, 2), padding='valid')(inputs)
    x = Conv2D(filters=3, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = Conv2D(filters=3, kernel_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # x = Conv2D(filters=3, kernel_size=(2, 2), strides=(2, 2), padding='valid')(inputs)
    x = Flatten()(x)
    outputs = Dense(3)(x)
    model = Model(inputs, outputs)
    model.summary()
    model.compile(loss = 'mse' , optimizer= 'rmsprop')
    return model


def main():
    model = build_model()


if __name__ == '__main__':
    main()
