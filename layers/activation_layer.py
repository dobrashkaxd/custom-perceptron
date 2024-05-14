from abc import abstractmethod
from layers.abstract_layer import Layer


class Activation(Layer):
    @abstractmethod
    def activation(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    def forward(self, x):
        """
        Performs the forward pass of the activation layer.

        Parameters:
            x: Input data for the forward pass.

        Returns:
            The result of applying the activation function to the input data.
        """
        self.input = x
        return self.activation(x)

    def backward(self, error):
        """
        Performs the backward pass of the activation layer.

        Parameters:
            error: The error gradient from the next layer.

        Returns:
            The error gradient for the current layer.
        """
        return error * self.gradient(self.input)
