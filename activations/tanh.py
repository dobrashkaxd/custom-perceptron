import numpy as np
from layers.activation_layer import Activation


class Tanh(Activation):
    def activation(self, x):
        return 2 / (1 + np.exp(-2 * x))

    def gradient(self, x):
        return 1 - self.activation(x) ** 2
