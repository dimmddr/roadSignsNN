from DoubledLayer import *

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])


# For the start I create only one neural net. I will add more later.
def init(alfa_=1, filter_size=(5, 5, 3), filters_count=10, pool_size=2, seed=16):
    global alfa
    global first_conv
    global first_outp
    # Size of window, 12x12 made from 48x48
    input_size = (12, 12, 3)
    first_conv = DoubledLayer(
            activation_func=sigmoid,
            activation_func_deriv=d_sigmoid,
            filters_size=filter_size + (filters_count,),
            pooling_size=pool_size,
            seed=seed)
    first_outp = FullConectionLayer(
            activation_func=sigmoid,
            activation_func_deriv=d_sigmoid,
            input_size=(input_size[0] - filter_size[0] + 1) * (input_size[1] - filter_size[1] + 1) * filters_count /
                       pool_size ** 2,
            output_size=1,
            seed=seed
    )
    alfa = alfa_


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Generators can be move out later, when I will be in need of speed
def prepare_roi(roi, window_size, step):
    res = roi[np.array([i for i in range(0, window_size[0], step)])]
    res = res[:, np.array([i for i in range(0, window_size[1], step)])]
    return res


def forward(input_image, window_size=(48, 48), step=4):
    Result = namedtuple('Result', ['roi', 'value'])
    res = []
    # TODO Проверить корректность вызова shape
    for y in range(input_image.shape[0] - window_size[0] + 1):
        for x in range(input_image.shape[1] - window_size[1] + 1):
            roi = input_image[y: y + window_size[0], x: x + window_size[0]]
            roi = prepare_roi(roi, window_size, step)
            conv_outp = first_conv.forward(roi)
            result = Result(Rectangle(x, y, x + window_size[0], y + window_size[1]), first_outp.forward(conv_outp))
            res.append(result)
    return res


# Here I used quadratic cost function, if I change cost function I would need to rewrite this function
def compute_covering(window, label):
    dx = min(window.xmax, label.xmax) - max(window.xmin, label.xmin)
    dy = min(window.ymax, label.ymax) - max(window.ymin, label.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy / ((label.xmax - label.xmin) * (label.ymax - label.ymin))
    else:
        return 0


def compute_output_error(answer, label, window, z, outp_layer, percent):
    covering_percent = compute_covering(window, label)
    correct_answer = int(covering_percent >= percent)
    return (correct_answer - answer) * outp_layer.activation_function_derivative(z)


def learning(x_in, lbl_in):
    forward_results = forward(x_in)
    # Convert tuple into named tuple
    lbl = Rectangle(lbl_in[0], lbl_in[1], lbl_in[2], lbl_in[3])
    for res in forward_results:
        sigma = compute_output_error(answer=res.value.a, label=lbl, window=res.roi, z=res.value.z,
                                     outp_layer=first_outp)
        w = first_outp.get_weights()
        z_conv = first_conv.activation_function_derivative(first_conv.get_z())
        # Проверить размерности
        sigma_conv = np.dot(w, [sigma]) * z_conv
        first_conv.learn()


def predict(x_in):
    raise NotImplemented
