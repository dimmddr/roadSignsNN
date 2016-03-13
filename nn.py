import os
import sys
import timeit
from collections import namedtuple

import numpy
import theano
from sklearn.cross_validation import train_test_split
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


def prepare_dataset(dataset, lbls):
    """ Prepare the dataset
    :param dataset: numpy 4D array with shape (images_number, image_height, image_width, image_layers)
    :param lbls: labels for images in [0, 1]
    """

    # noinspection PyIncorrectDocstring
    def shared_dataset(data_x, data_y, borrow=True):
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

    train_set_x, tmp_x, train_set_y, tmp_y = train_test_split(dataset, lbls, test_size=0.2)
    valid_set_x, test_set_x, valid_set_y, test_set_y = train_test_split(tmp_x, tmp_y, test_size=0.5)

    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
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
        # (28, 28) is the size of MNIST images.
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
            poolsize=(2, 2)
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
            n_in=3 * 4 * 4,
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

    def one_cycle(self, datasets, n_epochs):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]

        n_train_batches //= self.batch_size
        n_valid_batches //= self.batch_size
        n_test_batches //= self.batch_size

        # Full image size = (3, 523, 1025)
        # "Break" full image into subimages of size = (3, 48, 48) where only every 4th columns and 4th rows counts,
        # others pixels throw away for now. So, essentially there is image with size = (3, 12, 12)
        # With subimages window step = 1 pixel we have 1396584 subimages and that quite a much, so i decide did
        # step = 2 with step = 2 for rows and column we will have 1396584 / 4 = 349146 subimages.
        # I will try if that's few enough

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
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.85
        validation_frequency = min(n_train_batches, patience // 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(n_test_batches)
                            ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def learning(self, dataset, labels, n_epochs=200):
        print(dataset.shape)
        dataset_first = self.convert48to12(dataset)
        print(dataset_first.shape)
        size = 1000
        for i in range(1):
            # for i in range(dataset.shape[0] // size):
            datasets = prepare_dataset(dataset_first[i * size: i * size + size, :, :, :],
                                       labels[i * size: i * size + size])
            self.one_cycle(datasets, n_epochs)
            # datasets = prepare_dataset(
            #     dataset[size * dataset.shape[0]: size * dataset.shape[0] + dataset.shape[0] % size, :, :, :],
            #     labels[size * dataset.shape[0]: size * dataset.shape[0] + dataset.shape[0] % size])
            # self.one_cycle(datasets, n_epochs)

