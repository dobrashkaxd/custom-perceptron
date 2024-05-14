import math
import numpy as np
from losses.abstract_loss import Loss


class MSELoss(Loss):
    def __init__(self, stopping_criterion=0):
        self.stopping_criterion = stopping_criterion

    def loss(self, y_true, y_pred):
        self.error = np.mean(((y_true - y_pred) ** 2))
        self.error_gradient = self.gradient(y_true, y_pred)
        return self.error

    def early_stopping(self):
        return math.isclose(self.error, self.stopping_criterion) or (
            self.error < self.stopping_criterion
        )

    def gradient(self, y_true, y_pred):
        return y_pred - y_true

    def backward_propagation(self, layers, y_true, y_pred):
        output = self.error_gradient
        count = len(layers)
        for layer in reversed(layers):
            output = layer.backward(output)
            count -= 1
