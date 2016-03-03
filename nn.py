from collections import namedtuple

import numpy
import theano
from theano import tensor as T

import network

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
debug_mode = False

IMG_WIDTH = 1025
IMG_HEIGHT = 523
IMG_LAYERS = 3

SUB_IMG_WIDTH = 12
SUB_IMG_HEIGHT = 12
SUB_IMG_LAYERS = 3


class Network(object):
    def __init__(self, batch_size=500, filter_numbers=10, learning_rate=1, random_state=42):
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch

        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        self.rng = numpy.random.RandomState(random_state)
        self.batch_size = batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # Reshape matrix of rasterized images of shape (batch_size, 3, 12 * 12)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = self.x.reshape((batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (12-5+1 , 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, filter_numbers, 4, 4)
        self.layer0 = network.ConvPoolLayer(
            self.rng,
            input=layer0_input,
            image_shape=(batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
            filter_shape=(filter_numbers, SUB_IMG_LAYERS, 5, 5),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, filter_numbers * 4 * 4),
        # or (500, 10 * 4 * 4) = (500, 160) with the default values.
        layer1_input = self.layer0.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer1 = network.HiddenLayer(
            self.rng,
            input=layer1_input,
            n_in=3 * 4 * 4,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        # TODO: Найти способ сделать выход не 2, а 1, у меня все-таки бинарная классификация, с ней это получится
        self.layer2 = network.LogisticRegression(input=self.layer1.output, n_in=500, n_out=2)

        # the cost we minimize during training is the NLL of the model
        self.cost = self.layer2.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer2.params + self.layer2.params + self.layer1.params + self.layer0.params

        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        self.updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grads)
            ]

    def learning(self, datasets):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # Full image size = (3, 523, 1025)
        # "Break" full image into subimages of size = (3, 48, 48) where only every 4th columns and 4th rows counts,
        # others pixels throw away for now. So, essentially there is image with size = (3, 12, 12)
        # With subimages window step = 1 pixel we have 1396584 subimages and that quite a much, so i decide did step = 2
        # with step = 2 for rows and column we will have 1396584 / 4 = 349146 subimages. I will try if that's few enough

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [self.index],
            self.layer2.errors(self.y),
            givens={
                self.x: test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        validate_model = theano.function(
            [self.index],
            self.layer2.errors(self.y),
            givens={
                self.x: valid_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: valid_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        train_model = theano.function(
            [self.index],
            self.cost,
            updates=self.updates,
            givens={
                self.x: train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
