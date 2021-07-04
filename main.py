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
    activation1 = nnn.activation.Relu()
    dense2 = nnn.layer.Dense(n_inputs=3,n_neurons=3)
    activation2 = nnn.activation.Softmax()
    cce = nnn.loss.CategoricalCrossEntropy()

    # Forward pass
    output = dense1.forward(X)
    output = activation1.forward(output)
    output = dense2.forward(output)
    output = activation2.forward(output)

    # Print result
    print(output[:5])

    # Calculate the network's current loss
    loss = cce.calculate(output, y)

    # Print loss value
    print("Loss: ", loss)

    # Calculate accuracy
    predictions = np.argmax(output,axis=1)
    if len(y.shape) == 2: y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions == y)

    # Print accuracy
    print("Accuracy: ", accuracy)