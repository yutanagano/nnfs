from .activation import Softmax
from .layer import Input
from .loss import CategoricalCrossEntropy, SoftmaxWithCategoricalCrossEntropy


class Module:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None


    def add(self, layer):
        self.layers.append(layer)


    def set(self, *, loss, optimiser, accuracy):
        self.loss = loss
        self.optimiser = optimiser
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


    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        self.accuracy.init(y)

        for epoch in range(1, epochs+1):

            output = self.forward(X, training=True)

            data_loss, reg_loss = self.loss.calculate(
                output,
                y,
                include_regularisation=True
            )
            loss = data_loss + reg_loss

            predictions = self.output_layer_activation.predict(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimiser.pre_update_params()
            for layer in self.trainable_layers:
                self.optimiser.update_params(layer)
            self.optimiser.post_update_params()

            if not epoch % print_every:
                print(
                    f'epoch: {epoch}, '
                    f'acc: {accuracy:.3f}, '
                    f'loss: {loss:.3f}, '
                    f'(data_loss: {data_loss:.3f}, reg_loss: {reg_loss:.3f}), '
                    f'lr: {self.optimiser.current_learning_rate}'
                )
        
        if validation_data is not None:
            X_val, y_val = validation_data

            output = self.forward(X_val, training=False)

            loss = self.loss.calculate(output, y_val)

            predictions = self.output_layer_activation.predict(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(
                'validation, '
                f'acc: {accuracy:.3f}, '
                f'loss: {loss:.3f}'
            )