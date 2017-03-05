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


def nn_init(sizes, batch_sizes, filters, learning_rate=0.1):
    # size = (SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 4, SUB_IMG_WIDTH // 4)
    first_net = neural_net_init(
        sizes=sizes[NET_12],
        learning_rate=learning_rate,
        batch_sizes=batch_sizes[NET_12],
        nn_filter=filters[NET_12]
    )

    # SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 2, SUB_IMG_WIDTH // 2
    second_net = neural_net_init(
        sizes=sizes[NET_24],
        learning_rate=learning_rate,
        batch_sizes=batch_sizes[NET_24],
        nn_filter=filters[NET_24]
    )

    # SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH
    third_net = neural_net_init(
        sizes=sizes[NET_48],
        learning_rate=learning_rate,
        batch_sizes=batch_sizes[NET_48],
        nn_filter=filters[NET_48]
    )
    # size = (SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 4, SUB_IMG_WIDTH // 4)
    first_calibrate_net = neural_net_init(
        sizes=sizes[NET_12],
        learning_rate=learning_rate,
        batch_sizes=batch_sizes[NET_12],
        nn_filter=filters[NET_12]
    )

    # SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 2, SUB_IMG_WIDTH // 2
    second_calibrate_net = neural_net_init(
        sizes=sizes[NET_24],
        learning_rate=learning_rate,
        batch_sizes=batch_sizes[NET_24],
        nn_filter=filters[NET_24]
    )
    return [first_net, second_net, third_net, first_calibrate_net, second_calibrate_net]


def learning(train_set, dataset_path, lbl_train, neural_nets, nn_for_learn, indexes, debug=False):
    if nn_for_learn[NET_12]:
        if debug:
            print("First network learning")
        for i in range(0, indexes[NET_12]):
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
            neural_nets[NET_12].learning(dataset=convert48to12(images), labels=labels, debug_print=debug, n_epochs=5)

    if nn_for_learn[NET_24]:
        if debug:
            print("Second network learning")
        for i in range(indexes[NET_24]):
            all_images, all_labels, coordinates = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                         lbl_train[i], debug=debug)
            predicted_labels = neural_nets[NET_12].predict(all_images[:, :, :, 1::4, 1::4])
            images = all_images[predicted_labels == 1]
            labels = all_labels[predicted_labels == 1]
            neural_nets[NET_24].learning(dataset=convert48to24(images), labels=labels, debug_print=debug, n_epochs=10)

    if nn_for_learn[NET_48]:
        if debug:
            print("Second network learning")
        for i in range(indexes[NET_48]):
            all_images, all_labels, coordinates = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                         lbl_train[i], debug=debug)
            predicted_labels_1 = neural_nets[NET_12].predict(all_images[:, :, :, 1::4, 1::4])
            images = all_images[predicted_labels_1 == 1]
            predicted_labels_2 = neural_nets[NET_24].predict(images[:, :, 1::2, 1::2])
            images = images[predicted_labels_2 == 1]
            labels = all_labels[predicted_labels_1 == 1][predicted_labels_2 == 1]
            neural_nets[NET_48].learning(dataset=images, labels=labels, debug_print=debug, n_epochs=10)
