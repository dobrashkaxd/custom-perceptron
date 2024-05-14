import numpy as np
from layers.abstract_layer import Layer


class Dense(Layer):
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features)
        self.bias = np.zeros(out_features)
        self.weights_gradient = None
        self.bias_gradient = None
        self.input = None

    def forward(self, x):
        self.input = x
        output = np.matmul(x, self.weights) + self.bias
        return output

    def backward(self, error):
        self.weights_gradient = np.dot(self.input.transpose(), error)
        self.bias_gradient = np.sum(error, axis=0)

        prev_error = np.dot(error, self.weights.transpose())

        return prev_error
