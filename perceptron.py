class Perceptron:
    def __init__(self, layers=None, optimizer=None, loss=None):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.history = None

    def add_layer(self, layer):
        if self.layers == None:
            self.layers = []
        self.layers.append(layer)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss(self, loss):
        self.loss = loss

    def forward_propagation(self, x):
        output = x
        count = 1
        for layer in self.layers:
            output = layer.forward(output)
            count += 1
        return output

    def fit(self, epochs, featurs, target, report=True):
        for epoch in range(epochs):
            if report:
                print('epoch', epoch)
            output = self.forward_propagation(featurs)
            loss_value = self.loss.loss(target, output)
            stop = self.loss.early_stopping()
            if stop:
                return
            self.loss.backward_propagation(self.layers, target, output)
            self.optimizer.step(self.layers)
            if report:
                print('loss:', loss_value)
            if self.history == None:
                self.history = []
            self.history.append(loss_value)
        return self.history

    def predict(self, x):
        return self.forward_propagation(x)
