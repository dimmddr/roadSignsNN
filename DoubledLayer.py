from collections import namedtuple

import numpy as np


class DoubledLayer:
    def __init__(self, activation_func, activation_func_deriv, input_size, filters_size=(5, 5, 3 * 10), pooling_size=2,
                 alfa=1, seed=16):
        np.random.seed(seed)
        self.filters = np.random.uniform(size=filters_size)
        self.filters_updates = np.zeros_like(self.filters)
        self.biases = np.random.uniform(size=(filters_size[2]))  # One bias for every filter
        self.biases_updates = np.zeros_like(self.biases)
        self.activation_function = activation_func
        self.activation_function_derivative = activation_func_deriv
        self.pool_size = pooling_size
        self.conv_res = np.zeros(shape=(input_size[0] - filters_size[0] + 1,
                                        input_size[1] - filters_size[1] + 1,
                                        filters_size[2]))
        self.conv_z = np.zeros(shape=(input_size[0] - filters_size[0] + 1,
                                      input_size[1] - filters_size[1] + 1,
                                      filters_size[2]))
        self.pool_res = np.zeros(shape=((input_size[0] - filters_size[0] + 1) / pooling_size,
                                        (input_size[1] - filters_size[1] + 1) / pooling_size,
                                        filters_size[2]))
        self.pool_indexes = np.empty(shape=((input_size[0] - filters_size[0] + 1) / pooling_size,
                                            (input_size[1] - filters_size[1] + 1) / pooling_size,
                                            filters_size[2]), dtype=object)
        self.alfa = alfa
        self.debug = False

    def get_weights(self):
        if self.debug:
            print("Doubled layer. Get weights function")
        return self.filters

    # Terrible, terrible 3 nested loops
    def forward(self, input_data: "numpy array of input values"):
        if self.debug:
            print("Doubled layer. Forward function. Convolutional layer stage")
        # Convolutional layer
        for layer in range(self.conv_res.shape[2]):
            for i in range(self.conv_res.shape[0]):
                for ii in range(self.conv_res.shape[1]):
                    t = np.dot(input_data[i: i + self.filters.shape[0], ii: ii + self.filters.shape[1], layer % 3]
                               .reshape((1, self.filters[:, :, layer].size)),
                               self.filters[:, :, layer].reshape((self.filters[:, :, layer].size, 1))) + self.biases[
                            layer]
                    self.conv_z[i, ii, layer] = t
                    self.conv_res[i, ii, layer] = self.activation_function(t)

        if self.debug:
            print("Doubled layer. Forward function. Pooling layer stage")
        # I think I need move this to another function, too much for one function already
        # Pooling layer
        for layer in range(self.pool_res.shape[2]):
            for i in range(self.pool_res.shape[0]):
                for ii in range(self.pool_res.shape[1]):
                    self.pool_res[i, ii, layer] = np.amax(self.conv_res[
                                                          i * self.pool_size: i * self.pool_size + self.pool_size,
                                                          ii * self.pool_size: ii * self.pool_size + self.pool_size,
                                                          layer])
                    # argmax - find index of max element
                    # unravel_index transform flat index into 2-D index
                    t = np.unravel_index(np.argmax(self.conv_res[
                                                   i * self.pool_size: i * self.pool_size + self.pool_size,
                                                   ii * self.pool_size: ii * self.pool_size + self.pool_size,
                                                   layer]), (self.pool_size, self.pool_size))
                    # Transform slice index into full array index
                    self.pool_indexes[i, ii, layer] = (t[0] + i * self.pool_size, t[1] + ii * self.pool_size)
        return self.pool_res

    # Can throw NameError
    def get_convolutional_layer_result(self):
        if self.debug:
            print("Doubled layer. Get convolutional layer result function")
        return self.conv_res

    def get_z(self):
        if self.debug:
            print("Doubled layer. Get z function")
        return self.conv_z

    def learn(self, partial_sigma, input_data):
        if self.debug:
            print("Doubled layer. Learn function")
        sigma = np.zeros_like(self.conv_z)
        # Beyond good and evil. I cry with tears of blood when I see this code
        for layer in range(self.pool_indexes.shape[2]):
            for i in range(self.pool_indexes.shape[0]):
                for ii in range(self.pool_indexes.shape[1]):
                    sigma[self.pool_indexes[i, ii, layer][0], self.pool_indexes[i, ii, layer][1], layer] = \
                        partial_sigma[i, ii, layer]
        sigma = sigma * self.activation_function_derivative(self.conv_z)
        weights = np.zeros_like(self.filters_updates)
        for layer in range(sigma.shape[2]):
            for i in range(sigma.shape[0]):
                for ii in range(sigma.shape[1]):
                    weights[:, :, layer] += input_data[i: i + self.filters.shape[0], ii: ii + self.filters.shape[1],
                                            layer % 3] * sigma[i, ii, layer]
        self.add_updates(filters=weights, biases=sigma)

    def update(self):
        if self.debug:
            print("Doubled layer. Update function")
        self.filters += self.filters_updates
        self.biases += self.biases_updates
        self.filters_updates = np.zeros_like(self.filters_updates)
        self.biases_updates = np.zeros_like(self.biases_updates)

    def add_updates(self, filters, biases):
        if self.debug:
            print("Doubled layer. Add update function")
        self.filters += filters
        self.biases_updates += biases

    def set_debug(self):
        self.debug = True


class FullConectionLayer:
    def __init__(self, activation_func, activation_func_deriv, input_size, output_size, alfa=1, seed=16):
        np.random.seed(seed)
        self.weights = np.random.uniform(size=(output_size, input_size))
        self.weights_updates = np.zeros_like(self.weights)
        self.biases = np.random.uniform(size=output_size)
        self.biases_updates = np.zeros_like(self.biases)
        self.activation_function = activation_func
        self.activation_function_derivative = activation_func_deriv
        self.alfa = alfa
        self.debug = False

    # Attention! weights param must have same size as this layer weights
    def set_weights(self, weights):
        if self.debug:
            print("Full connection layer. Set weights function")
        self.weights = weights

    # feed forward
    def forward(self, input_data):
        if self.debug:
            print("Full connection layer. Forward function")
        Result = namedtuple('FullConnectionLayerResult', ['a', 'z'])
        # input data have several dimension, but here I need it in only one
        input_data = input_data.ravel()
        if 1 == len(input_data.shape):
            input_data = input_data.reshape(input_data.shape + (self.weights.shape[0],))
        z = np.dot(input_data, self.weights) + np.sum(self.biases)
        return Result(self.activation_function(z), z)

    def get_weights(self):
        if self.debug:
            print("Full connection layer. Get weights function")
        return self.weights

    def update(self):
        if self.debug:
            print("Full connection layer. Update function")
        self.weights += self.weights_updates
        self.biases += self.biases_updates
        self.weights_updates = np.zeros_like(self.weights_updates)
        self.biases_updates = np.zeros_like(self.biases_updates)

    def add_updates(self, weights, biases):
        if self.debug:
            print("Full connection layer. Add updates function")
        self.weights_updates += weights
        self.biases_updates += biases

    def set_debug(self):
        self.debug = True
