import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "D:\Datasets\GTZAN Dataset - Music Genre Classification\data.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

        # convert list into numpy arrays
        inputs = np.array(data["mfcc"])
        targets = np.array(data["labels"])

        return inputs, targets


if __name__ == "__main__":
    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets,
                                                                              test_size=0.3)

    # build the network archuitecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),  # hop_lenght

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),  # Rectified Linear Unit

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu"),  # 512/256/65 are the number of neurons

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu"),

        # output layer
        keras.layers.Dense(10, activation="softmax")  # Is 10 neurons because we have 10 musical genres
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # stochastic gradient descent
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # train network

    model.fit(inputs_train, targets_train,
              validation_data=(inputs_test, targets_test),
              epochs=50,
              batch_size=32)  # see in slide  # for use the whole dataset, configure the batch_size=1
