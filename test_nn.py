import csv
import datetime as dt
import os
import random

import numpy as np
import numpy.lib.recfunctions

import nn
import prepare_images

# Дополнить изображения нулями до 1025х523 (максимальный размер + 1)
# +1 нужен для того чтобы сделать значения нечетными, тогда после сверточного слоя мы получим массив с четным размером
# а значит он без осложнений пройдет через pooling layer, где я беру максимум из 2х2 квадрата
# TODO: Придумать как хранить информацию о соответствии результатов сверточного слоя и весов
# TODO: Реализовать обучение сверточного слоя


dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
annotation_path = dataset_path + 'allAnnotations.csv'
negatives_path = dataset_path + "negatives.dat"


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

        for i in range(0, ind):
            print(i)
            all_imgs, all_lbls = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'), lbl_train[i])
            imgs = [all_imgs[i] for i in range(len(all_imgs)) if all_lbls[i] == 1]
            lbls = all_lbls[all_lbls == 1]
            neg_imgs = [all_imgs[i] for i in range(len(all_imgs)) if all_lbls[i] != 1]
            random.seed(42)
            neg_imgs = random.sample(neg_imgs, len(imgs) * 100)  # 100 times more negative subimages then positive
            neg_lbls = np.zeros(shape=len(imgs))
            tmp = imgs + neg_imgs
            lbls = np.concatenate((lbls, neg_lbls))
            index = random.sample(range(len(tmp)), len(tmp))
            lbls = lbls[index]
            imgs = [tmp[i] for i in range(len(tmp))]
            # print(imgs.shape)
            # print(lbls.shape)
            net.learning(dataset=imgs, labels=lbls)

            # net.save_params()
            # net.load_params()
            #
            # print("Testing...")
            # test_img = train_set_without_negatives['Filename'][ind + 1:ind + 2]
            # lbl_test = (train_set_without_negatives[['Upper_left_corner_X', 'Upper_left_corner_Y',
            #                                          'Lower_right_corner_X', 'Lower_right_corner_Y']][ind + 1:ind + 2])
            # imgs, lbls = prepare_images.prepare(dataset_path + test_img[0].decode('utf8'), lbl_test[0])
            # y_pred = net.predict(imgs)
            # tmp = lbls - y_pred
            # tp = np.sum(y_pred == 1 and lbls == 1)
            # tn = np.sum(y_pred == 0 and lbls == 0)
            # fp = np.sum(tmp == -1)
            # fn = np.sum(tmp == 1)
            # print("True positive = {}, true negative = {}, false positive = {}, false negative = {}".format(tp, tn, fp, fn))

    return res


def test_all():
    print("Test learning speed")
    test_learning_speed(0.5, 3, 0.5, init=True)


if __name__ == '__main__':
    test_learning_speed()
