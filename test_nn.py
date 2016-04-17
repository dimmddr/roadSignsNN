import csv
import datetime as dt
import os

import numpy as np

import nn
import prepare_images
from image import Image

dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/"
annotation_path = dataset_path + 'frameAnnotations.csv'
# dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
# annotation_path = dataset_path + 'allAnnotations.csv'

# negatives_path = dataset_path + "negatives.dat"
# train_set_complete = np.empty(0)
train_set_without_negatives = dict()

NEGATIVE_MULTIPLIER = 1


def test_init(seed=16):
    # global train_set_complete
    global train_set_without_negatives
    np.random.seed(seed)

    # Read data from files
    image_data = np.genfromtxt(annotation_path, delimiter=';', names=True, dtype=None)
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
    # negatives = np.genfromtxt(negatives_path, dtype=None)
    # negatives = negatives.view(dtype=[('Filename', negatives.dtype.str)])
    # train_set_complete = numpy.lib.recfunctions.stack_arrays((image_data, negatives), autoconvert=True, usemask=False)
    train_set_without_negatives = files


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


# test for train speed
def show_some_weights(net):
    (w, b) = net.layer0_convPool.params
    # prepare_images.show_roi(w.get_value())
    print(w.get_value())


def test_learning_speed(min_speed=1., max_speed=2., step_size=1., init=False):
    # I don't want to do it multiply times, it is time costly to read large file
    if not init:
        test_init()

        # ind = int(np.floor(len(train_set_complete) * 0.75))
        ind = 10
        for alf in np.linspace(min_speed, max_speed, num=np.floor((max_speed - min_speed) / step_size)):
            print(alf)
            alfa = alf
            net = nn.Network(learning_rate=alfa, batch_size=5, random_state=123)
            train_set = np.array(list(train_set_without_negatives.keys()))
            train_set.sort()
            train_set = train_set[:ind]
            lbl_train = np.array([train_set_without_negatives.get(key).get_coordinates() for key in train_set])

            for i in range(0, ind):
                print(i)
                all_imgs, all_lbls = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'), lbl_train[i])
                print("Image prepared")
                imgs = all_imgs[all_lbls == 1]
                # prepare_images.show_roi(imgs)
                lbls = all_lbls[all_lbls == 1]
                # Set seed before every shuffle for consistent shuffle
                np.random.seed(42)
                np.random.shuffle(all_lbls)
                np.random.seed(42)
                np.random.shuffle(all_imgs)
                neg_size = int(lbls.shape[0] * NEGATIVE_MULTIPLIER)
                neg_lbls = all_lbls[:neg_size]
                neg_imgs = all_imgs[:neg_size]
                # print(lbls)
                imgs = np.concatenate((imgs, neg_imgs))
                lbls = np.concatenate((lbls, neg_lbls))
                # print(lbls)
                print(imgs.shape)
                print(lbls.shape)
                # Set seed before every shuffle for consistent shuffle
                np.random.seed(42)
                np.random.shuffle(lbls)
                np.random.seed(42)
                np.random.shuffle(imgs)
                net.learning(dataset=imgs, labels=lbls, debug_print=True)

            net.save_params()
            net.load_params()

            # show_some_weights(net)

            print("Testing...")
            test_img = np.array(list(train_set_without_negatives.keys())[:ind])
            lbl_test = np.array([train_set_without_negatives.get(key).get_coordinates() for key in test_img])
            imgs, lbls = prepare_images.prepare(dataset_path + test_img[0].decode('utf8'), lbl_test[0])
            y_pred = net.predict_values(imgs)
            tmp = lbls - y_pred

            tp = np.sum((y_pred == 1) & (lbls == 1))
            tn = np.sum((y_pred == 0) & (lbls == 0))
            fp = np.sum(tmp == -1)
            fn = np.sum(tmp == 1)
            f1_score = 2 * tp / (2 * tp + fn + fp)
            print("True positive = {}, true negative = {}, false positive = {}, false negative = {}\nf1 score = {}"
                  .format(tp, tn, fp, fn, f1_score))


def test_batch_size(min_size=5, max_size=200, step_size=5, init=False, debug=False):
    # I don't want to do it multiply times, it is time costly to read large file
    if not init:
        test_init()
        res = []

        # ind = int(np.floor(len(train_set_complete) * 0.75))
        ind = 20
        for batch_size in range(min_size, max_size, step_size):
            print("Batch size = {}".format(batch_size))
            alfa = 0.01
            filter_numbers = (100, 50)
            net = nn.Network(learning_rate=alfa, filter_numbers=filter_numbers, batch_size=batch_size, random_state=123)
            train_set = np.array(list(train_set_without_negatives.keys()))
            train_set.sort()
            train_set = train_set[:ind]
            lbl_train = np.array([train_set_without_negatives.get(key).get_coordinates() for key in train_set])

            for i in range(0, ind):
                if debug:
                    print(i)
                all_imgs, all_lbls, coords = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'),
                                                                    lbl_train[i], debug=debug)
                if debug:
                    print("Image prepared")
                imgs = all_imgs[all_lbls == 1]
                lbls = all_lbls[all_lbls == 1]
                neg_size = int(lbls.shape[0] * NEGATIVE_MULTIPLIER)
                neg_lbls = all_lbls[:neg_size]
                neg_imgs = all_imgs[:neg_size]
                imgs = np.concatenate((imgs, neg_imgs))
                lbls = np.concatenate((lbls, neg_lbls))
                net.learning(dataset=imgs, labels=lbls, debug_print=debug, n_epochs=200)

            net.save_params("test_batch_size_{}_filter_numbers_{}".format(batch_size, filter_numbers))
            net.load_params("test_batch_size_{}_filter_numbers_{}".format(batch_size, filter_numbers))

            # show_some_weights(net)

            print("Testing...")
            test_img = np.array(list(train_set_without_negatives.keys())[ind:])
            lbl_test = np.array([train_set_without_negatives.get(key).get_coordinates() for key in test_img])
            for i in range(test_img.shape[0]):
                imgs, lbls, coords = prepare_images.prepare(dataset_path + test_img[i].decode('utf8'), lbl_test[i])
                y_pred = net.predict(imgs)
                tmp = lbls - y_pred

                tp = np.sum((y_pred == 1) & (lbls == 1))
                tn = np.sum((y_pred == 0) & (lbls == 0))
                fp = np.sum(tmp == -1)
                fn = np.sum(tmp == 1)
                f1_score = 2 * tp / (2 * tp + fn + fp)
                print(" f1 score = {}, true positive = {}, true negative = {}, false positive = {}, false negative = {}"
                      .format(f1_score, tp, tn, fp, fn))
                res.append((batch_size, f1_score, tp, tn, fp, fn))
                tmp = y_pred == 1
                tmp = np.arange(lbls.shape[0])[tmp]
                rects = [coords.get(key, None) for key in tmp]
                prepare_images.show_rectangles(dataset_path + test_img[i].decode('utf8'), rects)
        return res


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


def test_all():
    # print("Test learning speed")
    # test_learning_speed(0.5, 3, 0.5, init=True)
    print("test batch size")
    res = test_batch_size(40, 50, 10, debug=True)
    print(res)
    # print("Test load parameters")
    # test_load_params()


if __name__ == '__main__':
    test_all()
