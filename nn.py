from DoubledLayer import *

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
debug_mode = False


# For the start I create only one neural net. I will add more later.
def init(alfa_=1, filter_size=(5, 5, 3), filters_count=10, pool_size=2, seed=16, debug=False):
    global alfa
    global first_conv
    global first_outp
    global debug_mode

    image_size = (523, 1025, 3)
    # Size of window, 12x12 made from 48x48
    input_size = (12, 12, 3)
    filter_size = (filter_size[0], filter_size[1], filter_size[2] * filters_count)
    first_conv = DoubledLayer(
            activation_func=sigmoid,
            activation_func_deriv=d_sigmoid,
            input_size=input_size,
            filters_size=filter_size,
            pooling_size=pool_size,
            seed=seed)
    first_outp = FullConectionLayer(
            activation_func=sigmoid,
            activation_func_deriv=d_sigmoid,
            input_size=(input_size[0] - filter_size[0] + 1) * (input_size[1] - filter_size[1] + 1) * filter_size[2] /
                       pool_size ** 2,
            output_size=1,
            seed=seed
    )
    alfa = alfa_
    if (debug):
        debug_mode = True
        first_conv.set_debug()
        first_outp.set_debug()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Generators can be move out later, when I will be in need of speed
def prepare_roi(roi, window_size, step):
    if debug_mode:
        print("Neural net. Prepare roi function")
    res = roi[np.array([i for i in range(0, window_size[0], step)])]
    res = res[:, np.array([i for i in range(0, window_size[1], step)])]
    return res


def forward(input_image):
    if debug_mode:
        print("Neural net. Forward function")
    conv_outp = first_conv.forward(input_image)
    result = (first_outp.forward(conv_outp), conv_outp)
    return result


# Here I used quadratic cost function, if I change cost function I would need to rewrite this function
def compute_covering(window, label):
    if debug_mode:
        print("Neural net. Compute covering function")
    dx = min(window.xmax, label.xmax) - max(window.xmin, label.xmin)
    dy = min(window.ymax, label.ymax) - max(window.ymin, label.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy / ((label.xmax - label.xmin) * (label.ymax - label.ymin))
    else:
        return 0


def compute_output_error(answer, label, window, z, outp_layer, percent):
    if debug_mode:
        print("Neural net. Compute output error function")
    covering_percent = compute_covering(window, label)
    correct_answer = int(covering_percent >= percent)
    return (correct_answer - answer) * outp_layer.activation_function_derivative(z)


def learning(x_in, lbl_in):
    if debug_mode:
        print("Neural net. Learning function")
    # TODO Придумать куда вынести эти параметры
    window_size = (48, 48)
    step = 4
    # Convert tuple into named tuple
    lbl = Rectangle(lbl_in[0], lbl_in[1], lbl_in[2], lbl_in[3])
    for y in range(x_in.shape[0] - window_size[0] + 1):
        for x in range(x_in.shape[1] - window_size[1] + 1):
            window = Rectangle(xmin=x, ymin=y, xmax=x + window_size[0], ymax=window_size[0])
            roi = x_in[window.ymin: window.ymax, window.xmin: window.xmax]
            roi = prepare_roi(roi, window_size, step)
            (forward_results, conv_outp) = forward(roi)
            sigma = compute_output_error(answer=forward_results.a, label=lbl, window=window, z=forward_results.z,
                                         outp_layer=first_outp)
            w = first_outp.get_weights()
            # Т.к. сигма в данном случае число, то матричное умножение вектора весов на сигму будет аналогично
            # поэлементному умножению весов на это число
            # Также надо отметить что я разделил формулу расчета сигмы и часть буду выполнять в методе класса двойного слоя
            partial_sigma_conv = w * sigma
            full_connection_biases_update = sigma
            full_connection_weights_update = conv_outp.ravel() * sigma
            first_outp.add_updates(full_connection_weights_update, full_connection_biases_update)
            first_conv.learn(partial_sigma_conv, x_in[window.xmin:window.xmax, window.ymin:window.ymax])
    print(np.amax(first_conv.conv_z))
    print(np.amin(first_conv.conv_z))
    first_outp.update()
    first_conv.update()


def predict(x_in):
    if debug_mode:
        print("Neural net. Predict function")
    raise NotImplemented
