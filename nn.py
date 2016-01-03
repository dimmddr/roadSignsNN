import numpy as np

input_layer_size = 28 * 28
hidden_layer_size = 15
output_layer_size = 10
alfa = 1
MAX_INPUT = 256

theta1 = (np.random.random((hidden_layer_size, input_layer_size + 1)) - 0.5) / 10
theta2 = (np.random.random((output_layer_size, hidden_layer_size + 1)) - 0.5) / 10


def init(input_size=28 * 28, hidden_size=15, output_size=10, alfa_=1, seed=16):
    global input_layer_size
    global hidden_layer_size
    global output_layer_size
    global alfa
    global theta1
    global theta2
    input_layer_size = input_size
    hidden_layer_size = hidden_size
    output_layer_size = output_size
    alfa = alfa_
    np.random.seed(seed)
    theta1 = (np.random.random((hidden_layer_size, input_layer_size + 1)) - 0.5) / 10
    theta2 = (np.random.random((output_layer_size, hidden_layer_size + 1)) - 0.5) / 10


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward(x):
    z_in = np.dot(theta1, x)
    z = sigmoid(z_in)
    z = np.insert(z, 0, 1)
    # forward, hidden to output
    y_in = np.dot(theta2, z)
    y = sigmoid(y_in)
    s = sum(y)
    y = y / s
    return {'y': y, 'y_in': y_in, 'z': z, 'z_in': z_in}


def learning(x_in, lbl_in):
    global theta1
    global theta2
    for i in range(len(x_in)):
        x = np.array(x_in[i]) / MAX_INPUT
        lbl = lbl_in[i]
        t = np.zeros(output_layer_size)
        t[lbl] = 1
        x = np.insert(x, 0, 1)
        # forward, input to hidden
        res = forward(x)
        y = res['y']
        y_in = res['y_in']
        z = res['z']
        z_in = res['z_in']

        # back, compute error for output
        sigma2 = (t - y) * d_sigmoid(y_in)
        d_theta2 = z[:, None] * sigma2 * alfa

        sigma_in = np.dot(theta2.T, sigma2)
        sigma1 = sigma_in[1:] * d_sigmoid(z_in)
        d_theta1 = x[:, None] * sigma1 * alfa
        theta1 = theta1 + d_theta1.T
        theta2 = theta2 + d_theta2.T


def predict(x_in):
    x = np.array(x_in) / MAX_INPUT
    x = np.insert(x, 0, 1)
    res = forward(x)['y']
    return np.argmax(res)
