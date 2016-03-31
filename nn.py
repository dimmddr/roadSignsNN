import pickle
import timeit
from collections import namedtuple

import numpy
import theano
from theano import tensor as T

import layers

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
debug_mode = False

IMG_WIDTH = 1025
IMG_HEIGHT = 523
IMG_LAYERS = 3

SUB_IMG_WIDTH = 12
SUB_IMG_HEIGHT = 12
SUB_IMG_LAYERS = 3


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


# TODO: Избавиться от магических чисел
class Network(object):
    def __init__(self, batch_size=500, filter_numbers=10, learning_rate=1, random_state=42):
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch

        self.x = T.tensor4('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        self.rng = numpy.random.RandomState(random_state)
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # Reshape matrix of rasterized images of shape (batch_size, 3, 12 * 12)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = self.x.reshape((batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (12-5+1 , 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, filter_numbers, 4, 4)
        self.layer0 = layers.ConvPoolLayer(
            self.rng,
            input=layer0_input,
            image_shape=(batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
            filter_shape=(filter_numbers, SUB_IMG_LAYERS, 5, 5),
            poolsize=(2, 2),
            activation_function="relu",
            relu_alpha=0.1
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, filter_numbers * 4 * 4),
        # or (500, 10 * 4 * 4) = (500, 160) with the default values.
        layer1_input = self.layer0.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer1 = layers.HiddenLayer(
            self.rng,
            input=layer1_input,
            n_in=10 * 4 * 4,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        # TODO: Найти способ сделать выход не 2, а 1, у меня все-таки бинарная классификация, с ней это получится
        self.layer2 = layers.LogisticRegression(input=self.layer1.output, n_in=500, n_out=2)

        # the cost we minimize during training is the NLL of the model
        self.cost = self.layer2.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer2.params + self.layer1.params + self.layer0.params

        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

    def convert48to12(self, dataset):
        return dataset[:, :, 1::4, 1::4]

    def one_cycle(self, datasets):
        train_set_x, train_set_y = datasets

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
            self.cost,
            updates=updates,
            givens={
                self.x: train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')

        start_time = timeit.default_timer()

        cost_ij = train_model(0)

        end_time = timeit.default_timer()
        # print(('The code for file ' +
        #        os.path.split(__file__)[1] +
        #        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def learning(self, dataset, labels, n_epochs=200):
        dataset_first = self.convert48to12(dataset)
        size = 1000
        for i in range(dataset.shape[0] // size):
            datasets = prepare_dataset(dataset_first[i * size: i * size + size, :, :, :],
                                       labels[i * size: i * size + size])
            self.one_cycle(datasets)
            # datasets = prepare_dataset(
            #     dataset[size * dataset.shape[0]: size * dataset.shape[0] + dataset.shape[0] % size, :, :, :],
            #     labels[size * dataset.shape[0]: size * dataset.shape[0] + dataset.shape[0] % size])
            # self.one_cycle(datasets, n_epochs)

    def predict(self, dataset):
        dataset_first = self.convert48to12(dataset)
        size = 500
        pred = theano.function(
            inputs=[self.layer0.input],
            outputs=self.layer2.y_pred
        )
        res = numpy.zeros(dataset.shape[0])
        for i in range(dataset.shape[0] // size):
            # datasets = prepare_dataset()
            res[i * size: i * size + size] = pred(dataset_first[i * size: i * size + size, :, :, :])
        return res

    def save_params(self):
        save_file = open('params', 'wb')
        pickle.dump(self.layer0.W, save_file, -1)
        pickle.dump(self.layer0.b, save_file, -1)
        pickle.dump(self.layer1.W, save_file, -1)
        pickle.dump(self.layer1.b, save_file, -1)
        pickle.dump(self.layer2.W, save_file, -1)
        pickle.dump(self.layer2.b, save_file, -1)
        save_file.close()

    def load_params(self):
        save_file = open('params', 'rb')
        self.layer0.W = pickle.load(save_file)
        self.layer0.b = pickle.load(save_file)
        self.layer1.W = pickle.load(save_file)
        self.layer1.b = pickle.load(save_file)
        self.layer2.W = pickle.load(save_file)
        self.layer2.b = pickle.load(save_file)
        save_file.close()
