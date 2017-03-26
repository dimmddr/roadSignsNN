import numpy as np

import prepare_images
from nn import Network, convert48to12, convert48to24
from settings import *

train_set_without_negatives = dict()
test_set_without_negatives = dict()


def neural_net_init(
        sizes,
        learning_rate,
        batch_sizes,
        nn_filter,
        pool_size=(2, 2),
        hidden_layer_size=500,
        softmax_units=2
):
    layers, height, width = sizes
    neural_net = Network(learning_rate=learning_rate,
                         input_shape=(batch_sizes, layers, height, width),
                         random_state=123)
    neural_net.add_convolution_layer(filter_numbers=nn_filter[0], filter_size=nn_filter[1])
    neural_net.add_pooling_layer(pool_size=pool_size)
    neural_net.add_dropout_layer(p=.5)
    neural_net.add_fully_connected_layer(hidden_layer_size=hidden_layer_size)
    neural_net.add_dropout_layer(p=.5)
    neural_net.add_softmax_layer(unit_numbers=softmax_units)
    neural_net.initialize()
    return neural_net


def nn_init(neural_nets_params, learning_rate=0.1):
    neural_nets = neural_nets_params.copy()
    neural_nets['net_12']['neural_net'] = neural_net_init(
        sizes=neural_nets['net_12']['sizes'],
        learning_rate=learning_rate,
        batch_sizes=neural_nets['net_12']['batch_size'],
        nn_filter=neural_nets['net_12']['filters']
    )

    neural_nets['net_24']['neural_net'] = neural_net_init(
        sizes=neural_nets['net_24']['sizes'],
        learning_rate=learning_rate,
        batch_sizes=neural_nets['net_24']['batch_size'],
        nn_filter=neural_nets['net_24']['filters']
    )

    neural_nets['net_48']['neural_net'] = neural_net_init(
        sizes=neural_nets['net_48']['sizes'],
        learning_rate=learning_rate,
        batch_sizes=neural_nets['net_48']['batch_size'],
        nn_filter=neural_nets['net_48']['filters']
    )

    neural_nets['calibration_net_24']['neural_net'] = neural_net_init(
        sizes=neural_nets['calibration_net_24']['sizes'],
        learning_rate=learning_rate,
        batch_sizes=neural_nets['calibration_net_24']['batch_size'],
        nn_filter=neural_nets['calibration_net_24']['filters']
    )

    neural_nets['calibration_net_48']['neural_net'] = neural_net_init(
        sizes=neural_nets['calibration_net_48']['sizes'],
        learning_rate=learning_rate,
        batch_sizes=neural_nets['calibration_net_48']['batch_size'],
        nn_filter=neural_nets['calibration_net_48']['filters']
    )
    return neural_nets


def learning_localization_networks(train_set, dataset_path, lbl_train, neural_nets, debug=False):
    if debug:
        print("First network learning")
    for i in range(0, neural_nets['net_12']['indexes']):
        if debug:
            print(i)
        all_images, all_labels, coordinates = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                     lbl_train[i], debug=debug)
        if debug:
            print("Image prepared")
        images = all_images[all_labels == 1]
        labels = np.ones(images.shape[0])
        neg_size = int(labels.shape[0] * NEGATIVE_MULTIPLIER)
        neg_indexes = np.random.choice(np.arange(all_images.shape[0] * all_images.shape[1]),
                                       neg_size, replace=False)
        neg_indexes = np.unravel_index(neg_indexes, (all_images.shape[0], all_images.shape[1]))
        neg_labels = all_labels[neg_indexes]
        neg_images = all_images[neg_indexes]
        images = np.concatenate((images, neg_images))
        labels = np.concatenate((labels, neg_labels))
        if debug:
            print("images.shape, labels.shape")
            print(images.shape, labels.shape)
            neural_nets['net_12']['neural_net'].learning(dataset=convert48to12(images), labels=labels,
                                                         debug_print=debug, n_epochs=5)

    if debug:
        print("Second network learning")
    for i in range(neural_nets['net_24']['indexes']):
        all_images, all_labels, coordinates = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                     lbl_train[i], debug=debug)
        predicted_labels = neural_nets['net_12']['neural_net'].predict(all_images[:, :, :, 1::4, 1::4])
        images = all_images[predicted_labels == 1]
        labels = all_labels[predicted_labels == 1]
        neural_nets['net_24']['neural_net'].learning(dataset=convert48to24(images), labels=labels, debug_print=debug,
                                                     n_epochs=10)

    if debug:
        print("Second network learning")
    for i in range(neural_nets['net_48']['indexes']):
        all_images, all_labels, coordinates = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                     lbl_train[i], debug=debug)
        predicted_labels_1 = neural_nets['net_12']['neural_net'].predict(all_images[:, :, :, 1::4, 1::4])
        images = all_images[predicted_labels_1 == 1]
        predicted_labels_2 = neural_nets['net_24']['neural_net'].predict(images[:, :, 1::2, 1::2])
        images = images[predicted_labels_2 == 1]
        labels = all_labels[predicted_labels_1 == 1][predicted_labels_2 == 1]
        neural_nets['net_48']['neural_net'].learning(dataset=images, labels=labels, debug_print=debug, n_epochs=10)


def learning_scale_network():
    pass
