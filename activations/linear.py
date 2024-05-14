from layers.activation_layer import Activation


class Linear(Activation):
    def __init__(self, bias=0):
        self.bias = bias

    def activation(self, x):
        """
        Computes the activation function on the input x.

        Parameters:
            x: Input data to apply the activation function.

        Returns:
            The result of applying the activation function to the input data.
        """
        return x + self.bias

    def gradient(self, x):
        return 1
