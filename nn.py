from DoubledLayer import *

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
debug_mode = False


# TODO Превратить это месиво в класс. Плевать что будет только один экземпляр класса, зато я замаскирую глобальные переменные под поля класса
# For the start I create only one neural net. I will add more later.
def init(alfa_=1, filter_size=(5, 5, 3), filters_count=10, pool_size=2, seed=16, debug=False):
    global alfa
    global first_conv
    global first_outp
    global debug_mode
    global window_size
    global stride
    global threshold

    image_size = (523, 1025, 3)
    window_size = (48, 48)
    stride = 4
    threshold = 12000
    # Max output:
    #   input in ~[0, 1], weights in [0, 1], every neuron is sum of 5*5*3 weighted inputs
    #   convolutional layer output is 8x8x10 array, pooling layer output is 4x4x10 array
    #   or 160 full connected layer input.
    #   Max of every convolutional output is 1 * 1 * 5 * 5 * 3 = 75
    #   Max of final output is 75 * 160 = 12000.
    #   Actually, input can be a little larger than 1, but it doesn't matter - if output bigger than threshold
    #   neural net will learn just fine.
    # I think I even can set threshold to 1 actually, but then weights will be very small in the end

    # Size of window, 12x12 made from 48x48
    input_size = (12, 12, 3)
    first_conv = DoubledLayer(
        activation_func=lrelu,
        activation_func_deriv=d_lrelu,
            input_size=input_size,
            filters_size=filter_size,
            filters_count=filters_count,
            pooling_size=pool_size,
            seed=seed)
    first_outp = FullConectionLayer(
        activation_func=lrelu,
        activation_func_deriv=d_lrelu,
            input_size=(input_size[0] - filter_size[0] + 1) * (input_size[1] - filter_size[1] + 1) * filters_count /
                       pool_size ** 2,
            output_size=1,
            seed=seed
    )
    alfa = alfa_
    if debug:
        debug_mode = True
        first_conv.set_debug()
        first_outp.set_debug()


def lrelu(x):
    return np.maximum(x, 0.01)


def d_lrelu(x):
    return (x > 0) + 0.01


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Generators can be move out later, when I will be in need of speed
def prepare_roi(roi, window):
    if debug_mode:
        print("Neural net. Prepare roi function")
    res = roi[np.array([i for i in range(0, window_size[0], stride)])]
    res = res[:, np.array([i for i in range(0, window_size[1], stride)])]
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
    if covering_percent >= percent:
        correct_answer = threshold
    else:
        correct_answer = 0
    return (correct_answer - answer) * outp_layer.activation_function_derivative(z)


def learning(x_in, lbl_in):
    if debug_mode:
        print("Neural net. Learning function")
    # Convert tuple into named tuple
    lbl = Rectangle(lbl_in[0], lbl_in[1], lbl_in[2], lbl_in[3])
    percent = 0.5
    cnt = 0
    x = 0
    y = 0
    numbers = 10000
    window = Rectangle(xmin=x, ymin=y, xmax=x + window_size[0], ymax=y + window_size[1])
    roi = x_in[window.xmin: window.xmax, window.ymin: window.ymax]
    # roi_time = timeit(lambda: prepare_roi(roi, window), number = numbers)
    # print("prepare_roi, average time={}, run {} times".format(roi_time / numbers, numbers))
    roi = prepare_roi(roi, window)
    (forward_results, conv_outp) = forward(roi)
    # forward_time = timeit(lambda: forward(roi), number=numbers)
    # print("forward, average time={}, run {} times".format(forward_time / numbers, numbers))
    sigma = compute_output_error(answer=forward_results.a, label=lbl, window=window, z=forward_results.z,
                                 outp_layer=first_outp, percent=percent)
    # compute_output_error_time = timeit(lambda: compute_output_error(answer=forward_results.a, label=lbl, window=window, z=forward_results.z,
    #                              outp_layer=first_outp, percent=percent), number=numbers)
    # print("compute_output_error, average time={}, run {} times".format(compute_output_error_time / numbers, numbers))
    w = first_outp.get_weights()
    partial_sigma_conv = w * sigma
    full_connection_biases_update = sigma
    full_connection_weights_update = conv_outp.ravel() * sigma
    first_outp.add_updates(full_connection_weights_update, full_connection_biases_update)
    first_conv.learn(partial_sigma_conv, roi)
    # learn_time = timeit(lambda: first_conv.learn(partial_sigma_conv, roi), number=numbers)
    # print("first_conv.learn, average time={}, run {} times".format(learn_time / numbers, numbers))

    # for x in range(x_in.shape[0] - window_size[0]):
    #     for y in range(x_in.shape[1] - window_size[1]):
    #         window = Rectangle(xmin=x, ymin=y, xmax=x + window_size[0], ymax=y + window_size[1])
    #         roi = x_in[window.xmin: window.xmax, window.ymin: window.ymax]
    #         roi = prepare_roi(roi, window)
    #         (forward_results, conv_outp) = forward(roi)
    #         sigma = compute_output_error(answer=forward_results.a, label=lbl, window=window, z=forward_results.z,
    #                                      outp_layer=first_outp, percent=percent)
    #         w = first_outp.get_weights()
    #         # Т.к. сигма в данном случае число, то матричное умножение вектора весов на сигму будет аналогично
    #         # поэлементному умножению весов на это число
    #         # Также надо отметить что я разделил формулу расчета сигмы
    #         # и часть буду выполнять в методе класса двойного слоя
    #         partial_sigma_conv = w * sigma
    #         full_connection_biases_update = sigma
    #         full_connection_weights_update = conv_outp.ravel() * sigma
    #         first_outp.add_updates(full_connection_weights_update, full_connection_biases_update)
    #         first_conv.learn(partial_sigma_conv, roi)
    #         cnt += 1
    #         if cnt % 1000 == 0:
    #             print(cnt)
    first_outp.update()
    first_conv.update()


def predict(x_in):
    if debug_mode:
        print("Neural net. Predict function")
    raise NotImplemented
