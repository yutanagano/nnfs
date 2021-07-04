import numpy as np


# Base class for loss functions
class Loss:

    # Calculates loss given model output and target/ground truth
    def calculate(self, pred : np.array, truth : np.array):

        # Calculate sample losses
        sample_losses = self.forward(pred, truth)

        # Calculate mean loss
        loss = np.mean(sample_losses)

        # Return mean loss
        return loss


# Categorial cross-entropy loss
class CategoricalCrossEntropy(Loss):

    # Forward pass
    def forward(self, pred : np.array, truth : np.array):

        # Get the number of samples in this batch
        num_samples = len(pred)

        # Clip to prevent exploding numbers
        pred_clipped = np.clip(pred, 1e-7, 1-1e-7)

        # If truth is provided as target values
        if len(truth.shape) == 1:
            loss = -np.log(
                pred_clipped[
                    range(num_samples),
                    truth
                ]
            )
        # If truth is provided as distributions
        else:
            loss = -np.sum(
                np.log(pred_clipped) * truth,
                axis=1
            )
        
        # Return the calculated loss
        return loss
