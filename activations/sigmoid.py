import numpy as np
from layers.activation_layer import Activation


class Sigmoid(Activation):
    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.activation(x) * (1 - self.activation(x))
