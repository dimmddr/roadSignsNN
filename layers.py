import lasagne
import numpy as np
from theano.tensor.signal.downsample import max_pool_2d

WIDTH_INDEX = 3
HEIGHT_INDEX = 2
LAYER_INDEX = 1


class SpatialPoolingLayer(lasagne.layers.Layer):
    # I assume that all bins has square shape for simplicity
    # Maybe later I change this behaviour
    def __init__(self, incoming, bin_sizes, **kwargs):
        super(SpatialPoolingLayer, self).__init__(incoming, **kwargs)
        self.bin_sizes = bin_sizes

    def get_output_shape_for(self, input_shape):
        tmp = np.array(self.bin_sizes)
        return np.sum(np.power(tmp, 2)),

    def get_output_for(self, input, **kwargs):
        layers = []
        for bin_size in self.bin_sizes:
            win_size = (np.ceil(input.shape[WIDTH_INDEX] / bin_size), np.ceil(input.shape[HEIGHT_INDEX] / bin_size))
            stride = (np.floor(input.shape[WIDTH_INDEX] / bin_size), np.floor(input.shape[HEIGHT_INDEX] / bin_size))
            layers.append(max_pool_2d(input=input, ds=win_size, st=stride))
        return lasagne.layers.concat(layers)
