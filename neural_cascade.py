import numpy as np

import prepare_images
from nn import Network, convert48to12, convert48to24
from settings import *

train_set_without_negatives = dict()
test_set_without_negatives = dict()


def nn_init(sizes, batch_sizes, filters, learning_rate=0.1):
    # size = (SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 4, SUB_IMG_WIDTH // 4)
    layers, height, width = sizes[NET_12]
    first_net = Network(learning_rate=learning_rate,
                        input_shape=(batch_sizes[NET_12], layers, height, width),
                        random_state=123
                        )
    first_filter = filters[0]
    first_net.add_convolution_layer(filter_numbers=first_filter[0], filter_size=first_filter[1])
    first_net.add_pooling_layer(pool_size=(2, 2))
    first_net.add_dropout_layer(p=.5)
    first_net.add_fully_connected_layer(hidden_layer_size=500)
    first_net.add_dropout_layer(p=.5)
    first_net.add_softmax_layer(unit_numbers=2)
    first_net.initialize()

    # SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 2, SUB_IMG_WIDTH // 2
    layers, height, width = sizes[NET_24]
    second_net = Network(learning_rate=learning_rate,
                         input_shape=(batch_sizes[NET_24], layers, height, width),
                         random_state=123
                         )
    second_filter = filters[1]
    second_net.add_convolution_layer(filter_numbers=second_filter[0], filter_size=second_filter[1])
    second_net.add_pooling_layer(pool_size=(2, 2))
    second_net.add_dropout_layer(p=.5)
    second_net.add_fully_connected_layer(hidden_layer_size=500)
    second_net.add_dropout_layer(p=.5)
    second_net.add_softmax_layer(unit_numbers=2)
    second_net.initialize()

    # SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH
    layers, height, width = sizes[NET_48]
    third_net = Network(learning_rate=learning_rate,
                        input_shape=(batch_sizes[NET_48], layers, height, width),
                        random_state=123
                        )
    third_filter = filters[2]
    third_net.add_convolution_layer(filter_numbers=third_filter[0], filter_size=third_filter[1])
    third_net.add_pooling_layer(pool_size=(2, 2))
    third_net.add_dropout_layer(p=.5)
    third_net.add_fully_connected_layer(hidden_layer_size=500)
    third_net.add_dropout_layer(p=.5)
    third_net.add_softmax_layer(unit_numbers=2)
    third_net.initialize()
    return [first_net, second_net, third_net]


def learning(train_set, dataset_path, lbl_train, neural_nets, nn_for_learn, indexes, debug=False):
    if nn_for_learn[NET_12]:
        if debug:
            print("First network learning")
        for i in range(0, indexes[NET_12]):
            if debug:
                print(i)
            all_images, all_labels = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
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
            all_images, all_labels = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                            lbl_train[i], debug=debug)
            predicted_labels = neural_nets[NET_12].predict(all_images)
            images = all_images[predicted_labels == 1]
            labels = all_labels[predicted_labels == 1]
            nn_for_learn[NET_24].learning(dataset=convert48to24(images), labels=labels, debug_print=debug, n_epochs=10)

    if nn_for_learn[NET_48]:
        if debug:
            print("Second network learning")
        for i in range(indexes[NET_48]):
            all_images, all_labels = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                            lbl_train[i], debug=debug)
            predicted_labels = neural_nets[NET_12].predict(all_images)
            images = all_images[predicted_labels == 1]
            predicted_labels = neural_nets[NET_24].predict(images)
            images = all_images[predicted_labels == 1]
            labels = all_labels[predicted_labels == 1]
            nn_for_learn[NET_48].learning(dataset=convert48to24(images), labels=labels, debug_print=debug, n_epochs=10)
