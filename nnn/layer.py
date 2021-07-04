import numpy as np


# Dense (fully connected) layer class
class Dense:

    # Constructor
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass function
    def forward(self, inputs : np.array):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

        return self.outputs

    # Backpropagation
    def backward(self, dvalues : np.array):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values to be passed back
        self.dinputs = np.dot(dvalues, self.weights.T)
