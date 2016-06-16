import numpy as np

import prepare_images
from nn import Network, convert48to12, convert48to24
from test_nn import dataset_path

SUB_IMG_WIDTH = 48
SUB_IMG_HEIGHT = 48
SUB_IMG_LAYERS = 3

train_set_without_negatives = dict()
test_set_without_negatives = dict()

NEGATIVE_MULTIPLIER = 2

first_ind = 75
second_ind = 250

first_batch_size = 50
second_batch_size = 30
third_batch_size = 30
alfa = 0.01
first_filter_numbers = 100
second_filter_numbers = 200
third_filter_numbers = 200
first_filter_size = (5, 5)
second_filter_size = (7, 7)
third_filter_size = (7, 7)


def nn_init():
    first_net = Network(learning_rate=alfa,
                        input_shape=(first_batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 4, SUB_IMG_WIDTH // 4),
                        random_state=123
                        )

    first_net.add_convolution_layer(filter_numbers=first_filter_numbers, filter_size=first_filter_size)
    first_net.add_pooling_layer(pool_size=(2, 2))
    first_net.add_dropout_layer(p=.5)
    first_net.add_fully_connected_layer(hidden_layer_size=500)
    first_net.add_dropout_layer(p=.5)
    first_net.add_softmax_layer(unit_numbers=2)
    first_net.initialize()

    second_net = Network(learning_rate=alfa,
                         input_shape=(
                             second_batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 2, SUB_IMG_WIDTH // 2),
                         random_state=123
                         )
    second_net.add_convolution_layer(filter_numbers=second_filter_numbers, filter_size=second_filter_size)
    second_net.add_pooling_layer(pool_size=(2, 2))
    second_net.add_dropout_layer(p=.5)
    second_net.add_fully_connected_layer(hidden_layer_size=500)
    second_net.add_dropout_layer(p=.5)
    second_net.add_softmax_layer(unit_numbers=2)
    second_net.initialize()

    third_net = Network(learning_rate=alfa,
                        input_shape=(
                            third_batch_size, SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
                        random_state=123
                        )
    third_net.add_convolution_layer(filter_numbers=third_filter_numbers, filter_size=third_filter_size)
    third_net.add_pooling_layer(pool_size=(2, 2))
    third_net.add_dropout_layer(p=.5)
    third_net.add_fully_connected_layer(hidden_layer_size=500)
    third_net.add_dropout_layer(p=.5)
    third_net.add_softmax_layer(unit_numbers=2)
    third_net.initialize()
    return [first_net, second_net, third_net]


def learning(train_set, lbl_train, neural_nets, nn_for_learn, indexes, debug=False):
    net_12, net_12_calibration, net_24, net_24_calibration, net_48 = (0, 1, 2, 3, 4)
    if nn_for_learn[net_12]:
        if debug:
            print("First network learning")
        for i in range(0, indexes[net_12]):
            if debug:
                print(i)
            all_imgs, all_lbls, coords = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                lbl_train[i], debug=debug)
            if debug:
                print("Image prepared")
            imgs = all_imgs[all_lbls == 1]
            lbls = np.ones(imgs.shape[0])
            neg_size = int(lbls.shape[0] * NEGATIVE_MULTIPLIER)
            neg_indexes = np.random.choice(np.arange(all_imgs.shape[0] * all_imgs.shape[1]),
                                           neg_size, replace=False)
            neg_indexes = np.unravel_index(neg_indexes, (all_imgs.shape[0], all_imgs.shape[1]))
            neg_lbls = all_lbls[neg_indexes]
            neg_imgs = all_imgs[neg_indexes]
            imgs = np.concatenate((imgs, neg_imgs))
            lbls = np.concatenate((lbls, neg_lbls))
            if debug:
                print("imgs.shape, lbls.shape")
                print(imgs.shape, lbls.shape)
            neural_nets[net_12].learning(dataset=convert48to12(imgs), labels=lbls, debug_print=debug, n_epochs=5)

    if nn_for_learn[net_12_calibration]:
        if debug:
            print("Second network learning")
        for i in range(indexes[net_12_calibration]):
            all_imgs, all_lbls = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                        lbl_train[i], debug=debug)
            nn_for_learn[net_12_calibration].learning(dataset=convert48to24(imgs), labels=lbls, debug_print=debug,
                                                      n_epochs=10)

    if nn_for_learn[2]:
        if debug:
            print("Third network learning")
        for i in range(second_ind, third_ind):
            if debug:
                print(i)
            all_imgs, all_lbls, coords = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                lbl_train[i], debug=debug)
            if debug:
                print("Image prepared")
            lbls = np.zeros_like(all_lbls)
            for j in range(all_imgs.shape[0]):
                lbls[j] = first_net.predict(convert48to12(all_imgs[j]))
            imgs = all_imgs[(lbls == 1) & (all_lbls == 1)]
            neg_size = int(imgs.shape[0] * NEGATIVE_MULTIPLIER)
            neg_imgs = all_imgs[(lbls == 1) & (all_lbls == 0)]
            lbls = np.ones(imgs.shape[0])
            if neg_imgs.shape[0] is not None and neg_imgs.shape[0] > 0:
                neg_indexes = np.random.choice(np.arange(neg_imgs.shape[0]), neg_size, replace=False)
                neg_imgs = neg_imgs[neg_indexes]

                neg_lbls = np.zeros(neg_imgs.shape[0])
                imgs = np.concatenate((imgs, neg_imgs))
                lbls = np.concatenate((lbls, neg_lbls))
            if debug:
                print("imgs.shape, lbls.shape")
                print(imgs.shape, lbls.shape)
            lbls = second_net.predict(dataset=convert48to24(imgs), labels=lbls, debug_print=debug, n_epochs=10)
            imgs = imgs[lbls == 1]
            third_net.learning(dataset=imgs, labels=lbls, debug_print=debug, n_epochs=10)
