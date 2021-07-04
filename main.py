import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from nnn import layers, activations

nnfs.init()

if __name__=="__main__":

    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create the necessary layers
    dense1 = layers.Dense(n_inputs=2,n_neurons=3)
    activation1 = activations.Relu()
    dense2 = layers.Dense(n_inputs=3,n_neurons=3)
    activation2 = activations.Softmax()

    # Forward pass
    output = dense1.forward(X)
    output = activation1.forward(output)
    output = dense2.forward(output)
    output = activation2.forward(output)

    # Print result
    print(output[:5])