import numpy as np


# Dense (fully connected) layer class
class Dense:

    # Constructor
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass function
    def forward(self, inputs : np.array):
        return np.dot(inputs, self.weights) + self.biases

