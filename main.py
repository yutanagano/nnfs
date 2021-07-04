import numpy as np
import nnfs
from nnfs.datasets import spiral_data

import nnn.layer, nnn.activation, nnn.loss

nnfs.init()

if __name__=="__main__":

    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create the necessary layers
    dense1 = nnn.layer.Dense(n_inputs=2,n_neurons=3)
    relu = nnn.activation.Relu()
    dense2 = nnn.layer.Dense(n_inputs=3,n_neurons=3)
    activation_loss = nnn.loss.SoftmaxWithCategoricalCrossentropy()

    # Forward pass
    output = dense1.forward(X)
    output = relu.forward(output)
    output = dense2.forward(output)

    # Calculate the network's current loss
    loss = activation_loss.forward(output, y)

    print(activation_loss.outputs[:5])

    # Print loss value
    print("Loss: ", loss)

    # Calculate accuracy
    predictions = np.argmax(output,axis=1)
    if len(y.shape) == 2: y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions == y)

    # Print accuracy
    print("Accuracy: ", accuracy)

    # Backward pass
    activation_loss.backward(activation_loss.outputs, y)
    dense2.backward(activation_loss.dinputs)
    relu.backward(dense2.dinputs)
    dense1.backward(relu.dinputs)

    # Print the gradients
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)