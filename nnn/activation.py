import numpy as np


# Rectified linuear unit activation function
class Relu:

    def forward(self, inputs : np.array):
        # Return the inputs with negative numbers corrected to 0
        return np.maximum(0, inputs)


# Softwmax activation function
class Softmax:

    def forward(self, inputs : np.array):
        # Exponentiate the input tensor element-wise
        output = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Return the normalised output (sample-wise normalisation)
        return output / np.sum(output, axis=1, keepdims=True)
