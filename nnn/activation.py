import numpy as np


# Rectified linuear unit activation function
class Relu:

    def forward(self, inputs : np.array):
        # Output is the inputs with negative numbers corrected to 0
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

        # Return calculated output
        return self.outputs

    # Backpropagation
    def backward(self, dvalues : np.array):
        # Copy the dvalues
        self.dinputs = dvalues.copy()

        # Zero the gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softwmax activation function
class Softmax:

    def forward(self, inputs : np.array):
        # Exponentiate the input tensor element-wise
        self.outputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Return the normalised output (sample-wise normalisation)
        return self.outputs / np.sum(self.outputs, axis=1, keepdims=True)

    # Backpropagation
    def backward(self, dvalues : np.array):

        # Create an empty array in the shape of dvalues
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for i, (output, dvalue) in enumerate(zip(self.outputs, dvalues)):
            # Flatten output array
            output = output.reshape(-1,1)
            # Calculate Jacobian matrix of output
            j_matrix = np.diagflat(output) - np.dot(output, output.T)
            # Calculate a sample-wise gradient and add it to the array of sample gradients
            self.dinputs[i] = np.dot(j_matrix, dvalue)


# Sigmoid activation
class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.outputs) * self.outputs


# Linear activation
class Linear:
    # Forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

        return inputs
    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()