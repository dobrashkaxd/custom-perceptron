from layers.activation_layer import Activation


class Linear(Activation):
    def __init__(self, bias=0):
        self.bias = bias

    def activation(self, x):
        return x + self.bias

    def gradient(self, x):
        return 1
