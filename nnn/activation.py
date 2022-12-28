import numpy as np


# Rectified linuear unit activation function
class Relu:
    def forward(self, inputs : np.array, training):
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


    def predict(self, outputs):
        return outputs


# Softwmax activation function
class Softmax:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
        keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
        keepdims=True)
        self.outputs = probabilities

        
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.outputs, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)


    # Calculate predictions for outputs
    def predict(self, outputs):
        return np.argmax(outputs, axis=1)


# Sigmoid activation
class Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs


    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.outputs) * self.outputs


    def predict(self, outputs):
        return (outputs > 0.5) * 1


# Linear activation
class Linear:
    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.outputs = inputs

        return inputs


    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()


    def predict(self, outputs):
        return outputs