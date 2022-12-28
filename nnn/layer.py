import numpy as np


class Input:
    def forward(self, inputs, training):
        self.outputs = inputs
        return inputs


# Dense (fully connected) layer class
class Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, l1w=0, l2w=0, l1b=0, l2b=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.l1w = l1w
        self.l2w = l2w
        self.l1b = l1b
        self.l2b = l2b


    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.outputs = np.dot(inputs, self.weights) + self.biases

        return self.outputs


    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradients on regularization
        # L1 on weights
        if self.l1w > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.l1w * dL1
        
        # L2 on weights
        if self.l2w > 0:
            self.dweights += 2 * self.l2w * \
            self.weights
        
        # L1 on biases
        if self.l1b > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.l1b * dL1
        
        # L2 on biases
        if self.l2b > 0:
            self.dbiases += 2 * self.l2b * \
            self.biases
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


    def get_parameters(self):
        return self.weights, self.biases


    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate


    def forward(self, inputs, training):
        self.binary_mask =\
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        if not training:
            self.outputs = inputs.copy()
        else:
            self.outputs = inputs * self.binary_mask
            
        return self.outputs


    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask