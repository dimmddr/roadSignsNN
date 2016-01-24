from collections import namedtuple

import numpy as np


class DoubledLayer:
    def __init__(self, activation_func, activation_func_deriv, input_size, filters_size=(5, 5, 3 * 10), pooling_size=2,
                 alfa=1, seed=16):
        np.random.RandomState(seed)
        self.filters = np.random.uniform(size=filters_size)
        self.activation_function = activation_func
        self.activation_function_derivative = activation_func_deriv
        self.pool_size = pooling_size
        self.conv_res = np.empty(shape=(input_size[0] - filters_size[0] + 1,
                                        input_size[1] - filters_size[1] + 1,
                                        filters_size[2]))
        self.conv_z = np.empty(shape=(input_size[0] - filters_size[0] + 1,
                                      input_size[1] - filters_size[1] + 1,
                                      filters_size[2]))
        self.pool_res = np.empty(shape=((input_size[0] - filters_size[0] + 1) / pooling_size,
                                        (input_size[1] - filters_size[1] + 1) / pooling_size,
                                        filters_size[2]))
        self.pool_indexes = np.empty(shape=((input_size[0] - filters_size[0] + 1) / pooling_size,
                                            (input_size[1] - filters_size[1] + 1) / pooling_size,
                                            filters_size[2]), dtype=object)
        self.alfa = alfa

    def get_weights(self):
        return self.filters

    # Terrible, terrible 3 nested loops
    def forward(self, input_data: "numpy array of input values"):
        # Convolutional layer
        for layer in range(len(self.conv_res[0, 0])):
            for i in range(len(self.conv_res)):
                for ii in range(len(self.conv_res[0])):
                    t = np.sum(input_data[i: i + self.filters.shape[0], ii: ii + self.filters.shape[1], layer % 3] *
                               self.filters[:, :, layer])
                    self.conv_z[i, ii, layer] = t
                    self.conv_res[i, ii, layer] = self.activation_function(t)

        # I think I need move this to another function, too much for one function already
        # Pooling layer
        for layer in range(self.pool_res.shape[2]):
            for i in range(self.pool_res.shape[0]):
                for ii in range(self.pool_res.shape[1]):
                    self.pool_res[i, ii, layer] = np.amax(self.pool_res[
                                                          i * self.pool_size: i * self.pool_size + self.pool_size,
                                                          ii * self.pool_size: ii * self.pool_size + self.pool_size,
                                                          layer])
                    t = np.unravel_index(np.argmax(self.pool_res[
                                                   i * self.pool_size: i * self.pool_size + self.pool_size,
                                                   ii * self.pool_size: ii * self.pool_size + self.pool_size,
                                                   layer]), (self.pool_size, self.pool_size))
                    self.pool_indexes[i, ii, layer] = (t[0] + i * self.pool_size, t[1] + ii * self.pool_size)
        return self.pool_res

    # Can throw NameError
    def get_convolutional_layer_result(self):
        return self.conv_res

    def get_z(self):
        return self.conv_z

    def learn(self, error):
        raise NotImplemented


class FullConectionLayer:
    def __init__(self, activation_func, activation_func_deriv, input_size, output_size=2, alfa=1, seed=16):
        np.random.RandomState(seed)
        self.weights = np.random.uniform(size=(output_size, input_size))
        self.activation_function = activation_func
        self.activation_function_derivative = activation_func_deriv
        self.alfa = alfa

    # Attention! weights param must have same size as this layer weights
    def set_weights(self, weights):
        self.weights = weights

    # feed forward
    def forward(self, input_data):
        Result = namedtuple('FullConnectionLayerResult', ['a', 'z'])
        z = np.dot(input_data, self.weights)
        return Result(self.activation_function(z), z)

    def get_weights(self):
        return self.weights
