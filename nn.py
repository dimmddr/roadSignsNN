from DoubledLayer import *


# For the start I create only one neural net. I will add more later.
def init(alfa_=1, seed=16):
    global alfa
    global first_conv
    global first_outp
    # first layer input is 12x12 window.
    # Actually, it's 48x48 window, but I take only every 4th pixel
    # 10 filters for start, 5x5 size each
    filters_count = 10
    filter_w = filter_h = 5
    pool_size = 2
    first_conv = DoubledLayer(
            activation_func=sigmoid,
            activation_func_deriv=d_sigmoid,
            filters_size=(filters_count, filter_w, filter_h),
            pooling_size=pool_size,
            seed=seed)
    first_outp = FullConectionLayer(
            activation_func=sigmoid,
            activation_func_deriv=d_sigmoid,
            input_size=(12 - filter_w + 1) * (12 - filter_h + 1) * filters_count / pool_size ** 2,
            output_size=1,
            seed=seed
    )
    alfa = alfa_


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# One roi at a time for now
def forward(x):
    t_res = first_conv.forward(x)
    return first_outp.forward(t_res)


# Here I used quadratic cost function, if I change cost function I would need to rewrite this function
def compute_output_error(actual_answer, right_answer, z, outp_layer):
    return (right_answer - actual_answer) * outp_layer.activation_function_derivative(z)


def learning(x_in, lbl_in):
    res = forward(x_in)
    sigma = compute_output_error(actual_answer=res[0], right_answer=lbl_in, z=res[1], outp_layer=first_outp)
    first_conv.learn()


def predict(x_in):
    raise NotImplemented
