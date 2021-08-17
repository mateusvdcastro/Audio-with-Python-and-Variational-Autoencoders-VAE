import numpy as np


class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        # num_hidden=[3, 5] -> this mean that we have 2 hidden layers, the first
        # contain 3 neurons and the secound contain 5 neurons;
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])  # creating random arrays with 2 dimensions
            self.weights.append(w)

    def forward_propagate(self, inputs):
        activations = inputs

        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w)  # dot() makes the dot product (produto escalar) entre matrizes ou vetores

            # calculate the activations
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # create an MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perfom forward propagation
    outputs = mlp.forward_propagate(inputs)

    # print the results
    print(f"The network input is: {inputs}")
    print(f"The network output is: {outputs}")
