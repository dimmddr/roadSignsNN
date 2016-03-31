import csv
import datetime as dt
import os

import numpy as np
import numpy.lib.recfunctions

import nn
import prepare_images

# TODO: Придумать как хранить информацию о соответствии результатов сверточного слоя и весов


dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
annotation_path = dataset_path + 'allAnnotations.csv'
negatives_path = dataset_path + "negatives.dat"
train_set_complete = np.empty(0)
train_set_without_negatives = np.empty(0)

# TODO: Try different numbers and see how it goes
NEGATIVE_MULTIPLIER = 40


def test_init(seed=16):
    global train_set_complete
    global train_set_without_negatives
    np.random.seed(seed)

    # Read data from files
    image_data = np.genfromtxt(annotation_path, delimiter=';', names=True, dtype=None)
    negatives = np.genfromtxt(negatives_path, dtype=None)
    negatives = negatives.view(dtype=[('Filename', negatives.dtype.str)])
    train_set_complete = numpy.lib.recfunctions.stack_arrays((image_data, negatives), autoconvert=True, usemask=False)
    train_set_without_negatives = image_data

    # Get random
    np.random.shuffle(train_set_complete)
    np.random.shuffle(train_set_without_negatives)


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
    (w, b) = net.layer0.params
    # Too much weights, don't want print all of them
    print("First filter weights:")
    print(w.get_value()[0])
    # But not so much biases, so why not
    print("All bias:")
    print(b.get_value())


def test_learning_speed(min_speed=1., max_speed=2., step_size=1., init=False):
    # I don't want to do it multiply times, read large file is long.
    if not init:
        test_init()

    res = []
    # ind = int(np.floor(len(train_set_complete) * 0.75))
    ind = 1
    for alf in np.linspace(min_speed, max_speed, num=np.floor((max_speed - min_speed) / step_size)):
        print(alf)
        alfa = alf
        net = nn.Network(learning_rate=alfa, random_state=123)
        train_set = train_set_complete['Filename'][0:ind]
        lbl_train = (train_set_complete[['Upper_left_corner_X', 'Upper_left_corner_Y',
                                         'Lower_right_corner_X', 'Lower_right_corner_Y']][0:ind])

        show_some_weights(net)

        for i in range(0, ind):
            print(i)
            all_imgs, all_lbls = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'), lbl_train[i])
            print("Image prepared")
            imgs = all_imgs[all_lbls == 1]
            lbls = all_lbls[all_lbls == 1]
            # Set seed before every shuffle for consistent shuffle
            np.random.seed(42)
            np.random.shuffle(all_lbls)
            np.random.seed(42)
            np.random.shuffle(all_imgs)
            neg_size = lbls.shape[0] * NEGATIVE_MULTIPLIER
            neg_lbls = all_lbls[:neg_size]
            neg_imgs = all_imgs[:neg_size]
            imgs = np.concatenate((imgs, neg_imgs))
            lbls = np.concatenate((lbls, neg_lbls))
            print(imgs.shape)
            print(lbls.shape)
            net.learning(dataset=imgs, labels=lbls)

        net.save_params()
        net.load_params()

        show_some_weights(net)

        print("Testing...")
        test_img = train_set_without_negatives['Filename'][ind + 1:ind + 2]
        lbl_test = (train_set_without_negatives[['Upper_left_corner_X', 'Upper_left_corner_Y',
                                                 'Lower_right_corner_X', 'Lower_right_corner_Y']][ind + 1:ind + 2])
        imgs, lbls = prepare_images.prepare(dataset_path + test_img[0].decode('utf8'), lbl_test[0])
        y_pred = net.predict(imgs)
        tmp = lbls - y_pred

        tp = np.sum((y_pred == 1) & (lbls == 1))
        tn = np.sum((y_pred == 0) & (lbls == 0))
        fp = np.sum(tmp == -1)
        fn = np.sum(tmp == 1)
        f1_score = 2 * tp / (2 * tp + fn + fp)
        print("True positive = {}, true negative = {}, false positive = {}, false negative = {}\nf1 score = {}"
              .format(tp, tn, fp, fn, f1_score))

    return res


def test_all():
    print("Test learning speed")
    test_learning_speed(0.5, 3, 0.5, init=True)


if __name__ == '__main__':
    test_learning_speed()
