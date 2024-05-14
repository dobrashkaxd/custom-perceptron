import numpy as np
from layers.activation_layer import Activation


class ReLU(Activation):
    def activation(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)
