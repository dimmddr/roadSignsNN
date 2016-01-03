class ConvolutionalNeuralNet:
    def __init__(self, output_layer_size):
        self.output_layer_size = output_layer_size

    def learn(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented

    def save_weight(self):
        raise NotImplemented

    def load_weight(self):
        raise NotImplemented
