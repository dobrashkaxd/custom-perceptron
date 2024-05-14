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
        self.input = x
        return self.activation(x)

    def backward(self, error):
        return error * self.gradient(self.input)
