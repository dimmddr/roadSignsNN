from DoubledLayer import *


# For the start I create only one neural net. I will add more later.
def init(alfa_=1, seed=16):
    global alfa
    global first_conv
    global first_outp
    # first layer input is 12x12 window.
    # This window will be created from 48x48 window, where only every 4th pixel matter
    # 10 filters for start, 5x5 size each
    filters_count = 10
    filter_w = filter_h = 5
    pool_size = 2
    first_conv = DoubledLayer(
            filters_size=(filters_count, filter_w, filter_h),
            pooling_size=pool_size,
            seed=seed)
    first_outp = FullConectionLayer(
            input_size=(12 - filter_w + 1) * (12 - filter_h + 1) * filters_count / pool_size ** 2,
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


def learning(x_in, lbl_in):
    raise NotImplemented


def predict(x_in):
    raise NotImplemented
