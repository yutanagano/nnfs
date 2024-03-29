from .activation import Softmax
from .layer import Input
from .loss import CategoricalCrossEntropy, SoftmaxWithCategoricalCrossEntropy
import copy
import numpy as np
import pickle


class Module:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None


    def add(self, layer):
        self.layers.append(layer)


    def set(self, *, loss=None, optimiser=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        
        if optimiser is not None:
            self.optimiser = optimiser

        if accuracy is not None:
            self.accuracy = accuracy


    def finalise(self):
        self.input_layer = Input()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            elif i == layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Softmax) and\
            isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output =\
                SoftmaxWithCategoricalCrossEntropy()


    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.outputs, training)
        
        # "layer" is now the last object from the list, return its output
        return layer.outputs


    def backward(self, outputs, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(outputs, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return

        self.loss.backward(outputs, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # If there is validation data passed,
        # set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1
            # For better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                # Dividing rounds down. If there are some remaining
                # data, but nor full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularisation_loss = \
                    self.loss.calculate(output, batch_y,
                    include_regularisation=True)

                loss = data_loss + regularisation_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predict(
                output)
                accuracy = self.accuracy.calculate(predictions,
                batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimiser.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimiser.update_params(layer)
                self.optimiser.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularisation_loss:.3f}), ' +
                    f'lr: {self.optimiser.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularisation_loss = \
            self.loss.calculate_accumulated(
            include_regularisation=True)
            epoch_loss = epoch_data_loss + epoch_regularisation_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
            f'acc: {epoch_accuracy:.3f}, ' +
            f'loss: {epoch_loss:.3f} (' +
            f'data_loss: {epoch_data_loss:.3f}, ' +
            f'reg_loss: {epoch_regularisation_loss:.3f}), ' +
            f'lr: {self.optimiser.current_learning_rate}')
        
            # If there is the validation data
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)
    

    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default value if batch size is not being set
        validation_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        
        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            # Otherwise slice a batch
            else:
                batch_X = X_val[
                step*batch_size:(step+1)*batch_size
                ]
                batch_y = y_val[
                step*batch_size:(step+1)*batch_size
                ]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)
            # Calculate the loss
            self.loss.calculate(output, batch_y)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predict(
            output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
        f'acc: {validation_accuracy:.3f}, ' +
        f'loss: {validation_loss:.3f}')


    def predict(self, X, *, batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):
            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X

            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)


    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters


    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)


    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)


    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))


    def save(self, path):
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('outputs', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'outputs', 'dinputs',
            'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)


    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model