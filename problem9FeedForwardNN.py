from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten


def build_model():
    inputs = Input((28, 28))
    x = Flatten()(inputs)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(16, activation='sigmoid')(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(3)(x)
    model = Model(inputs, outputs)
    model.summary()
    return model


def main():
    model = build_model()


if __name__ == '__main__':
    main()
