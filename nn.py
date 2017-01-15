import timeit
from collections import namedtuple

import lasagne
import numpy as np
import theano
from lasagne.regularization import regularize_network_params
from theano import tensor as T

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])

WIDTH_INDEX = 2
HEIGHT_INDEX = 1
LAYER_INDEX = 0

DEFAULT_BATCH_SIZE = 50


def convert48to12(dataset):
    return dataset[:, :, 1::4, 1::4]


def convert48to24(dataset):
    return dataset[:, :, 1::2, 1::2]


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class Network(object):
    def __init__(self, input_shape, learning_rate=0.01, random_state=42):
        self.input = T.tensor4('inputs')
        self.target = T.ivector('targets')
        self.learning_rate = learning_rate
        self.random_state = random_state
        if input_shape[0] is not None:
            self.batch_size = input_shape[0]
        else:
            self.batch_size = DEFAULT_BATCH_SIZE
        # Input layer
        self.network = lasagne.layers.InputLayer(
            shape=input_shape,
            input_var=self.input
        )

    # noinspection PyAttributeOutsideInit
    def initialize(self):
        self.prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target)
        self.loss = loss.mean()

        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            self.loss, self.params, learning_rate=self.learning_rate, momentum=0.9)

        self.train_fn = theano.function([self.input, self.target], loss, updates=self.updates,
                                        allow_input_downcast=True)
        outputs = T.argmax(self.prediction, axis=1)
        # self.predict_values = theano.function([self.input], self.prediction, allow_input_downcast=True)
        self.predict_values = theano.function([self.input], outputs, allow_input_downcast=True)

        self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.target)
        l1 = regularize_network_params(self.network, lasagne.regularization.l1)
        l2 = regularize_network_params(self.network, lasagne.regularization.l2)
        self.test_loss = self.test_loss.mean() + (l1 * 1e-4) + l2

        self.test_acc = T.mean(T.eq(T.argmax(self.test_prediction, axis=1), self.target), dtype=theano.config.floatX)
        self.val_fn = theano.function([self.input, self.target], [self.test_loss, self.test_acc],
                                      allow_input_downcast=True)

    def add_convolution_layer(self, filter_numbers, filter_size):
        self.network = lasagne.layers.Conv2DLayer(
            incoming=self.network,
            num_filters=filter_numbers,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()
        )

    def add_pooling_layer(self, pool_size):
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=pool_size)

    def add_dropout_layer(self, p):
        self.network = lasagne.layers.dropout(self.network, p=p)

    def add_fully_connected_layer(self, hidden_layer_size):
        self.network = lasagne.layers.DenseLayer(
            self.network,
            num_units=hidden_layer_size,
            nonlinearity=lasagne.nonlinearities.rectify
        )

    def add_softmax_layer(self, unit_numbers):
        self.network = lasagne.layers.DenseLayer(
            self.network,
            num_units=unit_numbers,
            nonlinearity=lasagne.nonlinearities.softmax
        )

    def learning(self, dataset, labels, n_epochs=200, debug_print=False):
        np.random.seed(self.random_state)
        np.random.shuffle(dataset)
        np.random.seed(self.random_state)
        np.random.shuffle(labels)
        validation_index = int(dataset.shape[0] * 0.6)
        test_index = validation_index + int(dataset.shape[0] * 0.2)
        train_set_x = dataset[:validation_index]
        train_set_y = labels[:validation_index]
        validation_set_x = dataset[validation_index:test_index]
        validation_set_y = labels[validation_index:test_index]
        test_set_x = dataset[test_index:]
        test_set_y = labels[test_index:]

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        for epoch in range(n_epochs):
            train_err = 0
            train_batches = 0
            start_time = timeit.default_timer()

            for batch in iterate_minibatches(train_set_x, train_set_y, self.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(validation_set_x, validation_set_y, self.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = self.val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            if debug_print:
                end_time = timeit.default_timer()

                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, n_epochs, end_time - start_time))

        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(test_set_x, test_set_y, self.batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        if debug_print:
            print("Final results:")
            if test_batches > 0:
                print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
                print("  test accuracy:\t\t{:.2f} %".format(
                    test_acc / test_batches * 100))

    def predict(self, dataset):
        size = self.batch_size
        shape = dataset.shape
        if len(shape) == 5:
            dataset = np.reshape(dataset, (shape[0] * shape[1], shape[2], shape[3], shape[4]))
        res = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0] // size):
            res[i * size: (i + 1) * size] = self.predict_values(dataset[i * size: (i + 1) * size, :, :, :])
        if len(shape) == 5:
            res = np.reshape(res, (shape[0], shape[1]))
        return res

    def save_params(self, filename):
        name = filename
        print(name)
        np.savez(name, *lasagne.layers.get_all_param_values(self.network))

    def load_params(self, filename):
        name = filename + '.npz'
        print(name)
        with np.load(name) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)