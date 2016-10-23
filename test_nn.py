import csv
import datetime as dt
import os
from functools import lru_cache

import numpy as np

import neural_cascade
import nn
import prepare_images
from image import Image
from settings import *


@lru_cache(maxsize=None)
def test_init(seed=16):
    np.random.seed(seed)

    # Read data from files
    # Read learning set
    image_data = np.genfromtxt(ANNOTATION_LEARNING_PATH, delimiter=';', names=True, dtype=None)
    files = dict()
    for image in image_data:
        filepath = image['Filename']
        if filepath not in files:
            img = Image(filepath)
            img.add_sign(label=image['Annotation_tag'],
                         coordinates=(image['Upper_left_corner_X'], image['Upper_left_corner_Y'],
                                      image['Lower_right_corner_X'], image['Lower_right_corner_Y']))
            files[filepath] = img
        else:
            files[filepath].add_sign(label=image['Annotation_tag'],
                                     coordinates=(image['Upper_left_corner_X'], image['Upper_left_corner_Y'],
                                                  image['Lower_right_corner_X'], image['Lower_right_corner_Y']))
    train_set_without_negatives = files

    # Read test set
    image_data = np.genfromtxt(ANNOTATION_TEST_PATH, delimiter=';', names=True, dtype=None)
    files = dict()
    for image in image_data:
        filepath = image['Filename']
        if filepath not in files:
            img = Image(filepath)
            img.add_sign(label=image['Annotation_tag'],
                         coordinates=(image['Upper_left_corner_X'], image['Upper_left_corner_Y'],
                                      image['Lower_right_corner_X'], image['Lower_right_corner_Y']))
            files[filepath] = img
        else:
            files[filepath].add_sign(label=image['Annotation_tag'],
                                     coordinates=(image['Upper_left_corner_X'], image['Upper_left_corner_Y'],
                                                  image['Lower_right_corner_X'], image['Lower_right_corner_Y']))
    test_set_without_negatives = files

    return {'train_set': train_set_without_negatives, 'test_set': test_set_without_negatives}


def write_results(result: list, test_name):
    if 0 == len(result):
        print("No results to write.")
        return
    # Maybe I need to make it into start of file for easy tuning
    wd = os.getcwd()
    result_folder = wd + "\\results"
    try:
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
    except OSError as e:
        print("Can't create folder, because of this: " + str(e))
        return

    name = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H-%M-%S')
    test_path = result_folder + "\\" + test_name
    try:
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        os.chdir(test_path)
    except OSError as e:
        print("Can't create folder, because of this: " + str(e))
        return
    try:
        with open(name + ".csv", "w") as outcsv:
            fieldnames = ['training_size', 'accuracy', 'learning_speed', 'hidden_layer_size', 'time']
            writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
            writer.writeheader()
            for res in result:
                writer.writerow(res)
    finally:
        os.chdir(wd)


# Can test only first n nets, cannot test something from the middle without first ones
def testing_results(neural_nets, nn_params, nn_for_test, test_set_without_negatives, numbers_of_test_imgs=10):
    first_net, second_net = neural_nets[:2]
    first_net.load_params(nn_params[0])

    if nn_for_test[1]:
        second_net.load_params(nn_params[1])

    print("Testing...")
    test_img = np.array(list(test_set_without_negatives.keys())[:numbers_of_test_imgs])
    lbl_test = np.array([test_set_without_negatives.get(key).get_coordinates() for key in test_img])

    tp_all = np.zeros(numbers_of_test_imgs)
    tn_all = np.zeros(numbers_of_test_imgs)
    fp_all = np.zeros(numbers_of_test_imgs)
    fn_all = np.zeros(numbers_of_test_imgs)
    f1_score_all = np.zeros(numbers_of_test_imgs)
    tn_percent_all = np.zeros(numbers_of_test_imgs)

    for i in range(test_img.shape[0]):
        imgs, lbls, coords = prepare_images.prepare(dataset_path + test_img[i].decode('utf8'), lbl_test[i])
        y_pred = np.zeros_like(lbls)
        for j in range(imgs.shape[0]):
            # TODO добавить nms в цепочку
            y_pred[j] = first_net.predict(nn.convert48to24(imgs[j]))
        if nn_for_test[1]:
            tmp = imgs[y_pred == 1]
            y_pred[y_pred == 1] = second_net.predict(tmp)

        tmp = lbls - y_pred

        tp = np.sum((y_pred == 1) & (lbls == 1))
        tn = np.sum((y_pred == 0) & (lbls == 0))
        fp = np.sum(tmp == -1)
        fn = np.sum(tmp == 1)
        f1_score = 2 * tp / (2 * tp + fn + fp)
        tp_all[i] = tp
        tn_all[i] = tn
        fp_all[i] = fp
        fn_all[i] = fn
        f1_score_all[i] = f1_score
        print(" f1 score = {}, true positive = {}, true negative = {}, false positive = {}, false negative = {}"
              .format(f1_score, tp, tn, fp, fn))
        tn_percent = tn / (tn + fp) * 100
        tn_percent_all[i] = tn_percent
        print("True negative percent from all negatives = {}".format(tn_percent))
        tmp = np.arange(lbls.shape[0] * lbls.shape[1]).reshape(lbls.shape)
        tmp = tmp[y_pred == 1]
        rects = [coords.get(key, None) for key in tmp]
        prepare_images.save_img_with_rectangles(dataset_path, test_img[i].decode('utf8'), rects)
        # prepare_images.show_rectangles(dataset_path + test_img[i].decode('utf8'), rects, show_type='opencv')
        # rects = prepare_images.nms(rects, 0.3)
        # prepare_images.show_rectangles(dataset_path + test_img[i].decode('utf8'), rects, show_type='opencv')
    print("f1 score = {}, true positive = {}, true negative = {}, false positive = {}, false negative = {}"
          .format(f1_score_all.mean(), tp_all.mean(), tn_all.mean(), fp_all.mean(), fn_all.mean()))
    print("True negative percent from all negatives = {}".format(tn_percent_all.mean()))
    return tp_all, tn_all, fp_all, fn_all, f1_score_all, tn_percent_all


def test_classification(seed=16, init=False):
    train_sets = test_init()
    train_set_without_negatives = train_sets['train_set']
    ind = int(np.floor(len(train_set_without_negatives) * 0.75))
    signs, labels, labels_dict = prepare_images.get_roi_from_images(train_set_without_negatives.values(),
                                                                    dataset_path)
    np.random.seed(seed)
    np.random.shuffle(signs)
    np.random.seed(seed)
    np.random.shuffle(labels)
    train_set_sign = signs[:ind]
    train_set_lbls = labels[:ind]
    test_set_sign = signs[ind:]
    test_set_lbls = labels[ind:]
    clf = nn.Clf(batch_size=50, output_size=len(labels_dict))
    clf.learning(train_set_sign, train_set_lbls)
    y_pred = clf.predict(test_set_sign)
    tmp = test_set_lbls - y_pred

    tp = np.sum((y_pred == 1) & (test_set_lbls == 1))
    tn = np.sum((y_pred == 0) & (test_set_lbls == 0))
    fp = np.sum(tmp == -1)
    fn = np.sum(tmp == 1)
    f1_score = 2 * tp / (2 * tp + fn + fp)
    print(" f1 score = {}, true positive = {}, true negative = {}, false positive = {}, false negative = {}"
          .format(f1_score, tp, tn, fp, fn))


def test_neural_net(indexes, batch_sizes, filters, sizes, init=False, debug=False):
    # TODO: make filter meaning again
    # I don't want to do it multiply times, it is time costly to read large file
    train_sets = test_init()
    train_set_without_negatives = train_sets['train_set']
    first_net, second_net = neural_cascade.nn_init(sizes=sizes, batch_sizes=batch_sizes, learning_rate=0.01)

    # ind = int(np.floor(len(list(train_set_without_negatives.keys())) * 0.75))
    # print("Total {} images for learning".format(ind))

    train_set = np.array(list(train_set_without_negatives.keys()))
    train_set.sort()
    # TODO: remove magic numbers
    train_set = train_set[:(indexes[0] + indexes[1])]
    lbl_train = np.array([train_set_without_negatives.get(key).get_coordinates() for key in train_set])

    neural_cascade.learning(train_set=train_set,
                            lbl_train=lbl_train,
                            neural_nets=(first_net, second_net),
                            nn_for_learn=[True, True],
                            indexes=indexes,
                            debug=debug)

    first_net.save_params("first_lvl_test_batch_size_{}_filter_numbers_{}_on_{}_image_learn_with_{}_filters_size"
                          .format(batch_sizes[0], filters[0][0], indexes[0], filters[0][1]))

    second_net.save_params("second_lvl_test_batch_size_{}_filter_numbers_{}_on_{}_image_learn_with_{}_filters_size"
                           .format(batch_sizes[1], filters[1][0], indexes[1], filters[1][1]))

    nn_params = ("first_lvl_test_batch_size_50_filter_numbers_100_on_75_image_learn_with_(5, 5)_filters_size",
                 "second_lvl_test_batch_size_30_filter_numbers_200_on_150_image_learn")
    testing_results((first_net, second_net), nn_params, (first_net, second_net),
                    test_set_without_negatives=train_sets['test_set'])


def test_load_params(batch_size=45, random_state=123, init=False):
    train_sets = test_init()
    train_set_without_negatives = train_sets['train_set']
    net = nn.Network(batch_size=batch_size, random_state=random_state)
    test_img = np.array(list(train_set_without_negatives.keys()))
    test_img.sort()
    lbl_test = np.array([train_set_without_negatives.get(key).get_coordinates() for key in test_img])
    for i in range(test_img.shape[0]):
        imgs, lbls = prepare_images.prepare(dataset_path + test_img[i].decode('utf8'), lbl_test[i])
        y_pred = net.predict_values(imgs)
        tmp = lbls - y_pred

        tp = np.sum((y_pred == 1) & (lbls == 1))
        tn = np.sum((y_pred == 0) & (lbls == 0))
        fp = np.sum(tmp == -1)
        fn = np.sum(tmp == 1)
        f1_score = 2 * tp / (2 * tp + fn + fp)
        print(" f1 score = {}, true positive = {}, true negative = {}, false positive = {}, false negative = {}"
              .format(f1_score, tp, tn, fp, fn))
