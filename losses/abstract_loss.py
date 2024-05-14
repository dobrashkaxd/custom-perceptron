from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass
