import numpy as np


class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy


    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy


    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None


    # Calculates precision value
    # based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

            
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Categorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

        
    # No initialization is needed
    def init(self, y):
        pass


    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y