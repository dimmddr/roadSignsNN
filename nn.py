import pickle
import timeit
from collections import namedtuple

import lasagne
import numpy
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
    :param dataset: numpy 4D array with shape (images_number, image_height, image_width, image_layers)
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
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        if data_y is not None:
            shared_y = theano.shared(numpy.asarray(data_y,
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


class Network(object):
    def __init__(self, batch_size=100, filter_numbers=10, filter_shape=(SUB_IMG_LAYERS, 7, 7), pool_size=(2, 2),
                 hidden_layer_size=500, learning_rate=1, random_state=42):
        # Input layer
        self.network = lasagne.layers.InputLayer(
            shape=(batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH)
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

        self.learning_rate = learning_rate


    def convert48to12(self, dataset):
        return dataset[:, :, 1::4, 1::4]

    def learning(self, dataset, labels, n_epochs=200, debug_print=False):
        dataset_first = self.convert48to12(dataset)
        datasets = prepare_dataset(dataset_first,
                                   labels)
        train_set_x, train_set_y = datasets
        # train_set_x, train_set_y = datasets[0]
        # valid_set_x, valid_set_y = datasets[1]
        # test_set_x, test_set_y = datasets[2]

        # Full image size = (3, 523, 1025)
        # "Break" full image into subimages of size = (3, 48, 48) where only every 4th columns and 4th rows counts,
        # others pixels throw away for now. So, essentially there is image with size = (3, 12, 12)
        # With subimages window step = 1 pixel we have 1396584 subimages and that quite a much, so i decide did
        # step = 2 with step = 2 for rows and column we will have 1396584 / 4 = 349146 subimages.
        # I will check if that's few enough

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grads)
            ]

        train_model = theano.function(
            [self.index],
            (self.cost, self.grads[0]),
            updates=updates,
            givens={
                self.x: train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // self.batch_size
        ###############
        # TRAIN MODEL #
        ###############
        print('... training')

        start_time = timeit.default_timer()

        for minibatch_index in range(n_train_batches):
            (cost_ij, grad_0) = train_model(minibatch_index)

            if debug_print:
                print("Cost = {}".format(cost_ij))
                print("Gradient:{}".format(grad_0[0]))
                print("Positive labels count = {}".format(
                    numpy.sum(labels[minibatch_index * self.batch_size: (minibatch_index + 1) * self.batch_size])
                ))
                # print(minibatch_index)
                # print(n_train_batches)
                # print(labels[minibatch_index * self.batch_size: (minibatch_index + 1) * self.batch_size])

        end_time = timeit.default_timer()
        # print(('The code for file ' +
        #        os.path.split(__file__)[1] +
        #        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def predict(self, dataset):
        dataset_first = self.convert48to12(dataset)
        size = self.batch_size
        pred = theano.function(
            inputs=[self.layer0_convPool.input],
            outputs=self.layer2_logRegr.y_pred
        )
        res = numpy.zeros(dataset.shape[0])
        for i in range(dataset.shape[0] // size):
            # datasets = prepare_dataset()
            res[i * size: i * size + size] = pred(dataset_first[i * size: i * size + size, :, :, :])
        return res

    def save_params(self):
        save_file = open('params', 'wb')
        pickle.dump(self.layer0_convPool.W, save_file, -1)
        pickle.dump(self.layer0_convPool.b, save_file, -1)
        pickle.dump(self.layer1_hidden.W, save_file, -1)
        pickle.dump(self.layer1_hidden.b, save_file, -1)
        pickle.dump(self.layer2_logRegr.W, save_file, -1)
        pickle.dump(self.layer2_logRegr.b, save_file, -1)
        save_file.close()

    def load_params(self):
        save_file = open('params', 'rb')
        self.layer0_convPool.W = pickle.load(save_file)
        self.layer0_convPool.b = pickle.load(save_file)
        self.layer1_hidden.W = pickle.load(save_file)
        self.layer1_hidden.b = pickle.load(save_file)
        self.layer2_logRegr.W = pickle.load(save_file)
        self.layer2_logRegr.b = pickle.load(save_file)
        save_file.close()

    def get_internal_state(self, dataset):
        dataset_first = self.convert48to12(dataset)
        size = self.batch_size
        pred = theano.function(
            inputs=[self.layer0_convPool.input],
            outputs=(
                self.layer1_hidden.input,
                self.layer2_logRegr.input,
                self.layer2_logRegr.dot_product,
                self.layer2_logRegr.p_y_given_x
            )
        )
        res = numpy.empty((dataset.shape[0], 4))
        # for i in range(dataset.shape[0] // size):
        for i in range(10):
            # datasets = prepare_dataset()
            # res[i * size: i * size + size] = pred(dataset_first[i * size: i * size + size, :, :, :])
            tmp = pred(dataset_first[i * size: i * size + size, :, :, :])

            print("layer1_hidden.input: {}".format(tmp[0][:5]))
            print("layer2_logRegr.input: {}".format(tmp[1][:5]))
            print("layer2_logRegr.dot_product: {}".format(tmp[2][:5]))
            print("layer2_logRegr.p_y_given_x: {}".format(tmp[3][:5]))
            # return res
