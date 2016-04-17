import timeit
from collections import namedtuple

import lasagne
import numpy as np
import theano
from theano import tensor as T

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])

SUB_IMG_WIDTH = 24
SUB_IMG_HEIGHT = 24
SUB_IMG_LAYERS = 3
WIDTH_INDEX = 2
HEIGHT_INDEX = 1
LAYER_INDEX = 0


def prepare_dataset(dataset, lbls=None):
    """ Prepare the dataset
    :param dataset: np 4D array with shape (images_number, image_height, image_width, image_layers)
    :param lbls: labels for images in [0, 1]
    :return tuple of theano.shared if lbls is not None, otherwise return single theano.shared
    """

    # noinspection PyIncorrectDocstring
    def shared_dataset(data_x, data_y=None, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        if data_y is not None:
            shared_y = theano.shared(np.asarray(data_y,
                                                dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')
        return shared_x, None

    train_set_x, train_set_y = shared_dataset(dataset, lbls)

    if train_set_y is not None:
        rval = (train_set_x, train_set_y)
    else:
        rval = train_set_x
    return rval


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
    def __init__(self, batch_size=100, filter_numbers=(25, 25), filter_shape_first_convlayer=(SUB_IMG_LAYERS, 5, 5),
                 filter_shape_second_convlayer=(SUB_IMG_LAYERS, 3, 3), pool_size=(2, 2), hidden_layer_size=500,
                 learning_rate=0.01, random_state=42):
        self.input = T.tensor4('inputs')
        self.target = T.ivector('targets')
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.batch_size = batch_size
        # Input layer
        self.network = lasagne.layers.InputLayer(
            shape=(batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
            input_var=self.input
        )

        # Сверточный слой, принимает регион исходного изображения размером 3х12х12
        self.network = lasagne.layers.Conv2DLayer(
            incoming=self.network,
            num_filters=filter_numbers[0],
            filter_size=(filter_shape_first_convlayer[1], filter_shape_first_convlayer[2]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()
        )

        # Poolling layer
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=pool_size)

        # Второй сверточный слой
        self.network = lasagne.layers.Conv2DLayer(
            incoming=self.network,
            num_filters=filter_numbers[1],
            filter_size=(filter_shape_second_convlayer[1], filter_shape_second_convlayer[2]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()
        )

        # Poolling layer
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=pool_size)

        # Fully-connected layer with 50% dropout
        self.network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(self.network, p=.5),
            num_units=hidden_layer_size,
            nonlinearity=lasagne.nonlinearities.rectify
        )

        # Softmax layer with dropout. 2 unit output
        self.network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(self.network, p=.5),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax
        )
        self.prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target)
        self.loss = loss.mean()

        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            self.loss, self.params, learning_rate=self.learning_rate, momentum=0.9)

        self.train_fn = theano.function([self.input, self.target], loss, updates=self.updates,
                                        allow_input_downcast=True)
        self.predict_values = theano.function([self.input], T.argmax(self.prediction, axis=1),
                                              allow_input_downcast=True)

        self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.target)
        self.test_loss = self.test_loss.mean()

        self.test_acc = T.mean(T.eq(T.argmax(self.test_prediction, axis=1), self.target), dtype=theano.config.floatX)
        self.val_fn = theano.function([self.input, self.target], [self.test_loss, self.test_acc],
                                      allow_input_downcast=True)

    def learning(self, dataset, labels, n_epochs=200, debug_print=False):
        dataset_first = convert48to24(dataset)
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_first)
        np.random.seed(self.random_state)
        np.random.shuffle(labels)
        validation_index = int(dataset_first.shape[0] * 0.6)
        test_index = validation_index + int(dataset_first.shape[0] * 0.2)
        train_set_x = dataset_first[:validation_index]
        train_set_y = labels[:validation_index]
        validation_set_x = dataset_first[validation_index:test_index]
        validation_set_y = labels[validation_index:test_index]
        test_set_x = dataset_first[test_index:]
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
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

    def predict(self, dataset):
        dataset_first = convert48to24(dataset)
        size = self.batch_size
        res = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0] // size):
            # datasets = prepare_dataset()
            res[i * size: i * size + size] = self.predict_values(dataset_first[i * size: i * size + size, :, :, :])
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
