from collections import namedtuple

import numpy as np


class DoubledLayer:
    def __init__(self, activation_func, activation_func_deriv, filters_size=(12, 12, 10), pooling_size=2, alfa=1,
                 input_size=(523, 1025, 3), seed=16):
        np.random.RandomState(seed)
        self.filters = np.random.uniform(size=filters_size)
        self.activation_function = activation_func
        self.activation_function_derivative = activation_func_deriv
        self.pool_size = pooling_size
        # Я обрезал все изображения до 640х480 - в финальной системе все равно предполагается один размер изображения
        # Плюс в каждом изображении ровно три слоя в данном сете - отсюда все магические цифры
        self.conv_res = np.empty(shape=input_size + (filters_size[2],))
        self.pool_res = np.empty(shape=(0, 0, 0))
        self.alfa = alfa

    def get_weights(self):
        return self.filters

    # Terrible, terrible 3 nested loops
    def forward(self, input_data: "numpy array of input values"):
        # Convolutional layer
        h = len(self.filters[0])
        w = len(self.filters[0, 0])
        # Always in this set, for now at least
        layers = 3
        for filter_number in range(len(self.filters)):
            for input_layer in range(layers):
                for i in range(len(input_data) - h):
                    for ii in range(len(input_data[0]) - w):
                        self.conv_res[i, ii, filter_number + filter_number * input_layer] = self.activation_function(
                                np.sum(input_data[i: i + h, ii: ii + w, input_layer] * self.filters[filter_number])
                        )

        # I think I need move this to another function, too much for one function already
        # Pooling layer
        self.pool_res = np.empty(shape=(
            len(self.conv_res),
            len(self.conv_res[0]) / self.pool_size,
            len(self.conv_res[0, 0]) / self.pool_size
        ))
        for conv_layer in range(self.conv_res):
            for i in range(self.conv_res[0] / self.pool_size):
                for ii in range(self.conv_res[0, 0] / self.pool_size):
                    self.pool_res[conv_layer, i, ii] = max(self.conv_res[
                                                           conv_layer,
                                                           i * self.pool_size: i * self.pool_size + self.pool_size,
                                                           ii * self.pool_size: ii * self.pool_size + self.pool_size])
        return self.pool_res

    # Can throw NameError
    def get_convolutional_layer_result(self):
        return self.conv_res

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

    # Maybe it's will be better to move this to class field and static method
    def cost_function(self, actual_answer, right_answer):
        return
