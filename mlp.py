import keras
from tensorflow.python.keras.backend import dtype


def build_mlp(input_shape):
    x = keras.layers.Input(input_shape, batch_size=1, dtype="float32")
    dense1 = keras.layers.Dense(128,activation="relu")(x)
    dense2 = keras.layers.Dense(32,activation="relu")(dense1)
    dense3 = keras.layers.Dense(16,activation="relu")(dense2)
    dense4 = keras.layers.Dense(16,activation="relu")(dense3)
    denseout = keras.layers.Dense(2, activation="softmax")(dense4)

    model = keras.models.Model(inputs=x, outputs=denseout)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


