import numpy as np
import nnfs
from nnfs.datasets import spiral_data

import nnn.layer, nnn.activation, nnn.loss, nnn.optimiser

nnfs.init()

if __name__=="__main__":

    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create the necessary layers
    dense1 = nnn.layer.Dense(n_inputs=2,n_neurons=64)
    relu = nnn.activation.Relu()
    dense2 = nnn.layer.Dense(n_inputs=64,n_neurons=3)
    activation_loss = nnn.loss.SoftmaxWithCategoricalCrossentropy()

    # Create the optimiser
    optimiser = nnn.optimiser.Adam(learning_rate=0.01, decay=0.00001)

    # Training loop
    for epoch in range(10001):
        # Forward pass
        output = dense1.forward(X)
        output = relu.forward(output)
        output = dense2.forward(output)

        # Calculate the network's current loss
        loss = activation_loss.forward(output, y)

        # Calculate accuracy
        predictions = np.argmax(output,axis=1)
        if len(y.shape) == 2: y = np.argmax(y,axis=1)
        accuracy = np.mean(predictions == y)

        # Print accuracy
        if not epoch % 100: print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimiser.current_learning_rate}")

        # Backward pass
        activation_loss.backward(activation_loss.outputs, y)
        dense2.backward(activation_loss.dinputs)
        relu.backward(dense2.dinputs)
        dense1.backward(relu.dinputs)

        # Update weights and biases
        optimiser.pre_update_params()
        optimiser.update_params(dense1)
        optimiser.update_params(dense2)
        optimiser.post_update_params()