import numpy as np
from nnn.activation import Softmax


# Base class for loss functions
class Loss:

    # Calculates loss given model output and target/ground truth
    def calculate(self, y_pred : np.array, y_true : np.array):

        # Calculate sample losses
        sample_losses = self.forward(y_pred, y_true)

        # Calculate mean loss
        loss = np.mean(sample_losses)

        # Return mean loss
        return loss


    def regularsiation_loss(self, layer):
        rl = 0

        if layer.l1w > 0:
            rl += layer.l1w * np.sum(np.abs(layer.weights))
        if layer.l1b > 0:
            rl += layer.l1b * np.sum(np.abs(layer.biases))
        
        if layer.l2w > 0:
            rl += layer.l2w * np.sum(layer.weights ** 2)
        if layer.l2b > 0:
            rl += layer.l2b * np.sum(layer.biases ** 2)
        
        return rl


# Categorial cross-entropy loss
class CategoricalCrossEntropy(Loss):

    # Forward pass
    def forward(self, y_pred : np.array, y_true : np.array):

        # Get the number of samples in this batch
        num_samples = len(y_pred)

        # Clip to prevent exploding numbers
        pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # If y_true is provided as target values
        if len(y_true.shape) == 1:
            loss = -np.log(
                pred_clipped[
                    range(num_samples),
                    y_true
                ]
            )
        # If y_true is provided as distributions
        else:
            loss = -np.sum(
                np.log(pred_clipped) * y_true,
                axis=1
            )
        
        # Return the calculated loss
        return loss

    # Backpropagation
    def backward(self, y_pred : np.array, y_true : np.array):
        # Number of samples
        samples = len(y_pred)
        # Number of labels in every sample
        labels = len(y_pred[0])

        # If labels are sparse, convert to one-hot vector form
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / y_pred
        # Normalise gradient
        self.dinputs = self.dinputs / samples


# Softmax activation combined with categorical cross-entropy
class SoftmaxWithCategoricalCrossentropy:

    # Constructor
    def __init__(self):
        # Creates activation and loss function objects
        self.softmax = Softmax()
        self.categoricalcrossentropy = CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs : np.array, y_true : np.array):
        # Pass inputs through activation and set self.outputs variable
        self.outputs = self.softmax.forward(inputs)
        # Return the loss value
        return self.categoricalcrossentropy.calculate(self.outputs, y_true)

    # Backpropagation
    def backward(self, y_pred : np.array, y_true : np.array):
        
        # Number of samples
        samples = len(y_pred)

        # If labels are one-hot encoded, convert to spare form
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy the y_pred so we can modify
        self.dinputs = y_pred.copy()

        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # Normalise gradient
        self.dinputs = self.dinputs / samples


# Binary cross-entropy loss
class BinaryCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
        (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
        (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Squared Error loss
class MeanSquaredError(Loss):
    # L2 loss
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Absolute Error loss
class MeanAbsoluteError(Loss):
    # L1 loss
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples