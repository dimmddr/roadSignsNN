import lasagne
import numpy as np
from theano import tensor as T

WIDTH_INDEX = 3
HEIGHT_INDEX = 2
LAYER_INDEX = 1


class SpatialPoolingLayer(lasagne.layers.Layer):
    # I assume that all bins has square shape for simplicity
    # Maybe later I change this behaviour
    def __init__(self, incoming, bin_sizes, **kwargs):
        super(SpatialPoolingLayer, self).__init__(incoming, **kwargs)
        self.bin_sizes = self.add_param(np.array(bin_sizes), (len(bin_sizes),), name="bin_sizes")

    def get_output_shape_for(self, input_shape):
        return T.sum(T.power(self.bin_sizes, 2))

    def get_output_for(self, input, **kwargs):
        layers = []
        for bin_size in self.bin_sizes:
            win_size = (np.ceil(input.shape[WIDTH_INDEX] / bin_size), np.ceil(input.shape[HEIGHT_INDEX] / bin_size))
            stride = (np.floor(input.shape[WIDTH_INDEX] / bin_size), np.floor(input.shape[HEIGHT_INDEX] / bin_size))
            layers.append(lasagne.layers.flatten(
                lasagne.layers.MaxPool2DLayer(input, pool_size=win_size, stride=stride)
            ))
        return lasagne.layers.concat(layers)
