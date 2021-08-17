import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf


def generate_dataset(num_samples, test_size):
    x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])  # array([0.1, 0.2], [0.3, 0.4])
    y = np.array([[i[0] + i[1]] for i in x])  # array([0.3], [0.7])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)  # test_size=0.3 -> the test set will
    return x_train, x_test, y_train, y_test                                   # have 30% of the whole dataset


if __name__ == "__main__":
    x_train, x_test, y_train, y_test  = generate_dataset(5000, 0.3)
    print(f"x_test: \n {x_test}")  # notice the sum arithmetic done
    print(f"y_test: \n {y_test}")

    # build model: input layer 2 neurons -> 5 neurons 1 hidden_layer -> 1 output
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # compile model
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimiser, loss="MSE")

    # train model
    model.fit(x_train, y_train, epochs=100)  # model.fit(x_train, y_train, batch_size=1, epochs=100)

    # evaluate model
    print("\nModel evaluation:")
    model.evaluate(x_test, y_test, verbose=1)

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    print("\nSome predictions:")
    for d, p in zip(data, predictions):
        print(f"{d[0]} + {d[1]} = {p[0]}")
