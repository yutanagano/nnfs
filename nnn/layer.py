import numpy as np


# Dense (fully connected) layer class
class Dense:

    # Constructor
    def __init__(self, n_inputs, n_neurons, l1w=0, l1b=0, l2w=0, l2b=0):
        # Initialise weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularisation strength
        self.l1w = l1w
        self.l1b = l1b
        self.l2w = l2w
        self.l2b = l2b

    # Forward pass function
    def forward(self, inputs : np.array):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

        return self.outputs

    # Backpropagation
    def backward(self, dvalues : np.array):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        if self.l1w > 0:
            dl1w = np.ones_like(self.weights)
            dl1w[self.weights < 0] = -1
            self.dweights += self.l1w * dl1w
        if self.l2w > 0:
            self.dweights += 2 * self.l2w * self.weights

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.l1b > 0:
            dl1b = np.ones_like(self.biases)
            dl1b[self.biases < 0] = -1
            self.dbiases += self.l1b * dl1b
        if self.l2b > 0:
            self.dbiases += 2 * self.l2b * self.biases

        # Gradient on values to be passed back
        self.dinputs = np.dot(dvalues, self.weights.T)


class Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate


    def forward(self, inputs):
        self.binary_mask =\
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.outputs = inputs * self.binary_mask
        return self.outputs


    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask