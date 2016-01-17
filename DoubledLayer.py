import numpy as np

class DoubledLayer:
    def __init__(self, activation_func, activation_func_deriv, filters_size=(10, 5, 5), pooling_size=2, alfa=1,
                 seed=16):
        np.random.RandomState(seed)
        self.filters = np.random.uniform(size=filters_size)
        self.activation_function = activation_func
        self.activation_function_derivative = activation_func_deriv
        self.pool_size = pooling_size
        self.conv_res = np.empty(shape=(0, 0, 0))
        self.pool_res = np.empty(shape=(0, 0, 0))
        self.alfa = alfa

    def get_weights(self):
        return self.filters

    # Terrible, terrible 3 nested loops
    def forward(self, input_data: "numpy array of input values"):
        # Convolutional layer
        h = len(self.filters[0])
        w = len(self.filters[0, 0])
        # It's bad, but only here I know size of input
        self.conv_res = np.empty(shape=(len(self.filters),
                                        len(input_data) - h + 1,
                                        len(input_data[0]) - w + 1))
        for filter_number in range(self.filters):
            for i in range(len(input_data) - h + 1):
                for ii in range(len(input_data[0]) - w + 1):
                    self.conv_res[filter_number, i, ii] = self.activation_function(
                            sum(input_data[i: i + h, ii: ii + w] * self.filters[filter_number])
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
        z = np.dot(input_data, self.weights)
        return self.activation_function(z), z

    def get_weights(self):
        return self.weights

    # Maybe it's will be better to move this to class field and static method
    def cost_function(self, actual_answer, right_answer):
        return
