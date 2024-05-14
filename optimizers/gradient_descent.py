class OptimizerGD:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def step(self, layers):
    for layer in layers:
      if hasattr(layer, 'weights_gradient'):
        layer.weights -= self.learning_rate * layer.weights_gradient