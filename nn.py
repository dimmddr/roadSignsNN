import timeit
from collections import namedtuple

import lasagne
import numpy as np
import theano
from theano import tensor as T

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])

SUB_IMG_WIDTH = 12
SUB_IMG_HEIGHT = 12
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
    def __init__(self, batch_size=100, filter_numbers=10, filter_shape=(SUB_IMG_LAYERS, 7, 7), pool_size=(2, 2),
                 hidden_layer_size=500, learning_rate=0.01, random_state=42):
        self.input = T.tensor4('inputs')
        self.target = T.ivector('targets')
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(random_state)
        self.batch_size = batch_size
        # Input layer
        self.network = lasagne.layers.InputLayer(
            shape=(batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
            input_var=self.input
        )

        # Сверточный слой, принимает регион исходного изображения размером 3х12х12
        self.network = lasagne.layers.Conv2DLayer(
            incoming=self.network,
            num_filters=filter_numbers,
            filter_size=(filter_shape[1], filter_shape[2]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()
        )

        # Polling layer
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
        self.predict = theano.function([self.input], self.prediction, allow_input_downcast=True)

    def learning(self, dataset, labels, n_epochs=200, debug_print=False):
        dataset_first = convert48to12(dataset)
        datasets = prepare_dataset(dataset_first,
                                   labels)
        train_set_x, train_set_y = (dataset_first, labels)
        # train_set_x, train_set_y = datasets

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        for epoch in range(n_epochs):
            train_err = 0
            train_batches = 0
            start_time = timeit.default_timer()

            for batch in iterate_minibatches(train_set_x, train_set_y, self.batch_size, shuffle=False):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            end_time = timeit.default_timer()

            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, end_time - start_time))

    def predict(self, dataset):
        dataset_first = convert48to12(dataset)
        size = self.batch_size
        res = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0] // size):
            # datasets = prepare_dataset()
            res[i * size: i * size + size] = self.pred(dataset_first[i * size: i * size + size, :, :, :])
        return res

        # def save_params(self):
        #     save_file = open('params', 'wb')
        #     pickle.dump(self.layer0_convPool.W, save_file, -1)
        #     pickle.dump(self.layer0_convPool.b, save_file, -1)
        #     pickle.dump(self.layer1_hidden.W, save_file, -1)
        #     pickle.dump(self.layer1_hidden.b, save_file, -1)
        #     pickle.dump(self.layer2_logRegr.W, save_file, -1)
        #     pickle.dump(self.layer2_logRegr.b, save_file, -1)
        #     save_file.close()
        #
        # def load_params(self):
        #     save_file = open('params', 'rb')
        #     self.layer0_convPool.W = pickle.load(save_file)
        #     self.layer0_convPool.b = pickle.load(save_file)
        #     self.layer1_hidden.W = pickle.load(save_file)
        #     self.layer1_hidden.b = pickle.load(save_file)
        #     self.layer2_logRegr.W = pickle.load(save_file)
        #     self.layer2_logRegr.b = pickle.load(save_file)
        #     save_file.close()
        #
        # def get_internal_state(self, dataset):
        #     dataset_first = convert48to12(dataset)
        #     size = self.batch_size
        #     pred = theano.function(
        #         inputs=[self.layer0_convPool.input],
        #         outputs=(
        #             self.layer1_hidden.input,
        #             self.layer2_logRegr.input,
        #             self.layer2_logRegr.dot_product,
        #             self.layer2_logRegr.p_y_given_x
        #         )
        #     )
        #     res = np.empty((dataset.shape[0], 4))
        #     # for i in range(dataset.shape[0] // size):
        #     for i in range(10):
        #         # datasets = prepare_dataset()
        #         # res[i * size: i * size + size] = pred(dataset_first[i * size: i * size + size, :, :, :])
        #         tmp = pred(dataset_first[i * size: i * size + size, :, :, :])
        #
        #         print("layer1_hidden.input: {}".format(tmp[0][:5]))
        #         print("layer2_logRegr.input: {}".format(tmp[1][:5]))
        #         print("layer2_logRegr.dot_product: {}".format(tmp[2][:5]))
        #         print("layer2_logRegr.p_y_given_x: {}".format(tmp[3][:5]))
        #         # return res
