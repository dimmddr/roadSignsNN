import csv
import datetime as dt
import gc
import os

import numpy as np

import data_utils
import nn
import prepare_images
from image import Image

SUB_IMG_WIDTH = 48
SUB_IMG_HEIGHT = 48
SUB_IMG_LAYERS = 3

# dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
# annotation_path = dataset_path + 'allAnnotations.csv'
# dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/"
# annotation_path = dataset_path + 'frameAnnotations.csv'
dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
annotation_learning_path = dataset_path + 'learningAnnotations.csv'
annotation_test_path = dataset_path + 'testAnnotations.csv'

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


def test_init(seed=16):
    # global train_set_complete
    global train_set_without_negatives
    global test_set_without_negatives
    np.random.seed(seed)

    # Read data from files
    # Read learning set
    image_data = np.genfromtxt(annotation_learning_path, delimiter=';', names=True, dtype=None)
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
    image_data = np.genfromtxt(annotation_test_path, delimiter=';', names=True, dtype=None)
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


def nn_init():
    first_net = nn.Network(learning_rate=alfa,
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

    second_net = nn.Network(learning_rate=alfa,
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

    third_net = nn.Network(learning_rate=alfa,
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
            neural_nets[net_12].learning(dataset=nn.convert48to12(imgs), labels=lbls, debug_print=debug, n_epochs=5)

    if nn_for_learn[net_12_calibration]:
        if debug:
            print("Second network learning")
        for i in range(indexes[net_12_calibration]):
            all_imgs, all_lbls = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                        lbl_train[i], debug=debug)
            nn_for_learn[net_12_calibration].learning(dataset=nn.convert48to24(imgs), labels=lbls, debug_print=debug,
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
                lbls[j] = first_net.predict(nn.convert48to12(all_imgs[j]))
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
            lbls = second_net.predict(dataset=nn.convert48to24(imgs), labels=lbls, debug_print=debug, n_epochs=10)
            imgs = imgs[lbls == 1]
            third_net.learning(dataset=imgs, labels=lbls, debug_print=debug, n_epochs=10)


# Can test only first n nets, cannot test something from the middle without first ones
def testing_results(neural_nets, nn_params, nn_for_test, numbers_of_test_imgs=10):
    first_net, second_net = neural_nets
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
    if not init:
        test_init()
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


def test_neural_net(init=False, debug=False):
    # I don't want to do it multiply times, it is time costly to read large file
    if not init:
        test_init()
        first_net, second_net = nn_init()

        # ind = int(np.floor(len(list(train_set_without_negatives.keys())) * 0.75))
        # print("Total {} images for learning".format(ind))

        train_set = np.array(list(train_set_without_negatives.keys()))
        train_set.sort()
        train_set = train_set[:(first_ind + second_ind)]
        lbl_train = np.array([train_set_without_negatives.get(key).get_coordinates() for key in train_set])

        learning(train_set=train_set, lbl_train=lbl_train,
                 neural_nets=(first_net, second_net),
                 nn_for_learn=[True, True],
                 debug=debug)

        first_net.save_params("first_lvl_test_batch_size_{}_filter_numbers_{}_on_{}_image_learn_with_{}_filters_size"
                              .format(first_batch_size, first_filter_numbers, first_ind, first_filter_size))

        second_net.save_params("second_lvl_test_batch_size_{}_filter_numbers_{}_on_{}_image_learn_with_{}_filters_size"
                               .format(second_batch_size, second_filter_numbers, second_ind, second_filter_size))

        nn_params = ("first_lvl_test_batch_size_50_filter_numbers_100_on_75_image_learn_with_(5, 5)_filters_size",
                     "second_lvl_test_batch_size_30_filter_numbers_200_on_150_image_learn")
        testing_results((first_net, second_net), nn_params)


def test_load_params(batch_size=45, random_state=123, init=False):
    if not init:
        test_init()
    net = nn.Network(batch_size=batch_size, random_state=random_state)
    net.load_params()
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


# TODO расширить тест для сетей кроме первой или переименовать
def test_neural_net_learning_size(start_size, end_size, step, init=False, debug=False):
    # I don't want to do it multiply times, it is time costly to read large file
    if not init:
        test_init()

    with open("test_learning_size_result.txt", "w") as test_outp:
        # TODO: избавиться от магического числа
        outp_csv = np.empty(shape=(7,))
        outp_csv.fill(np.nan)
        for index in range(start_size, end_size, step):
            gc.collect()
            first_net, second_net = nn_init()
            print("Learning size = {}".format(index))
            first_ind = index
            train_set = np.array(list(train_set_without_negatives.keys()))
            train_set.sort()
            train_set = train_set[:(first_ind + second_ind)]
            lbl_train = np.array([train_set_without_negatives.get(key).get_coordinates() for key in train_set])

            nn_for_learn = [True, False]
            learning(train_set=train_set, lbl_train=lbl_train,
                     neural_nets=(first_net, second_net),
                     nn_for_learn=nn_for_learn,
                     indexes=(first_ind, second_ind),
                     debug=debug)

            nn_params = ('first_lvl_test_learning_size_{}_filter_numbers_{}_on_{}_image_learn_with_{}_filters_size'
                         .format(first_ind, first_filter_numbers, first_ind, first_filter_size),
                         'second_lvl_test_learning_size_{}_filter_numbers_{}_on_{}_image_learn_with_{}_filters_size'
                         .format(second_ind, second_filter_numbers, second_ind, second_filter_size)
                         )

            if nn_for_learn[0]:
                first_net.save_params(nn_params[0])

            if nn_for_learn[1]:
                second_net.save_params(nn_params[1])

            res = testing_results((first_net, second_net), nn_params, nn_for_test=[True, False])
            tp_all, tn_all, fp_all, fn_all, f1_score_all, tn_percent_all = res

            test_outp.write("Learning size = {}\n".format(index))
            test_outp.write(
                "f1 score = {}, true positive = {}, true negative = {}, false positive = {}, false negative = {}\n"
                    .format(f1_score_all.mean(), tp_all.mean(), tn_all.mean(), fp_all.mean(), fn_all.mean()))
            test_outp.write("True negative percent from all negatives = {}\n".format(tn_percent_all.mean()))
            index_outp = np.zeros_like(tp_all)
            index_outp.fill(index)
            tmp = np.column_stack((index_outp, tp_all, tn_all, fp_all, fn_all, f1_score_all, tn_percent_all))
            if outp_csv[0] is not np.nan:
                outp_csv = np.vstack((outp_csv, tmp))
            else:
                outp_csv = tmp

        np.savetxt('test_learning_size.csv', outp_csv)


def data_analysis():
    data_utils.analyse_sign_frame_size_fluctuations(annotation_learning_path, "sign_frame_size_fluctuations")


def test_all():
    # print("test learning size")
    # test_neural_net_learning_size(start_size=20, end_size=100, step=5, debug=True)
    # print("test batch size")
    # res = test_neural_net(debug=True)
    # print(res)
    # print("Test load parameters")
    # test_load_params()
    # print("Test Classification")
    # test_classification()
    data_analysis()


if __name__ == '__main__':
    test_all()
