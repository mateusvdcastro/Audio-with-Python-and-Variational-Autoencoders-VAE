import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "D:\Datasets\GTZAN Dataset - Music Genre Classification\data.json"


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

        # convert list into numpy arrays
        x = np.array(data["mfcc"])
        y = np.array(data["labels"])

        return x, y


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_dataset(test_size, validation_size):
    # load data
    x, y = load_data(DATA_PATH)

    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # create train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)


    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape):
    # create RNN-LSTM model
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, x, y):
    x = x[np.newaxis, ...]
    # prediction = [[0.1, 0.2, ...]]
    prediction = model.predict(x)  # X -> (130, 13, 1)
    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)  # [4]
    print(f"Expected index: {y}. Predicted index {predicted_index}")


if __name__ == "__main__":
    # create train, validation and test sets
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)

    # build the RNN-LSTM net
    input_shape = (x_train.shape[1], x_train.shape[2])   # 130, 13
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train the LSMT
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                        batch_size=32, epochs=30)

    plot_history(history)

    # evaluate the LSTM on the test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Accuracy on test set is: {test_accuracy}")

    # make prediction on a sample
    x = x_test[80]
    y = y_test[80]
    predict(model, x, y)
