import csv
import datetime as dt
import os
import timeit

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
    np.random.seed(seed)

    # Read data from files
    image_data = np.genfromtxt(annotation_path, delimiter=';', names=True, dtype=None)
    negatives = np.genfromtxt(negatives_path, dtype=None)
    negatives = negatives.view(dtype=[('Filename', negatives.dtype.str)])
    train_set_complete = numpy.lib.recfunctions.stack_arrays((image_data, negatives), autoconvert=True, usemask=False)

    # Get random
    np.random.shuffle(train_set_complete)


def test():
    image_data = np.genfromtxt(annotation_path, delimiter=';', names=True, dtype=None)
    negatives = np.genfromtxt(negatives_path, dtype=None)
    negatives = negatives.view(dtype=[('Filename', negatives.dtype.str)])
    train_set_complete = numpy.lib.recfunctions.stack_arrays((image_data, negatives), autoconvert=True, usemask=False)
    # img = prepare_images.prepare(dataset_path + train_set_complete[0]['Filename'].decode('utf8'))
    res = timeit.timeit(lambda: prepare_images.prepare(dataset_path + train_set_complete[0]['Filename'].decode('utf8')),
                        number=1000)
    print(res / 1000)


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
def test_learning_speed(min_speed=1., max_speed=2., step_size=1., batch_size=10, init=False):
    # I don't want to do it multiply times, read large file is long.
    if not init:
        test_init()

    res = []
    # ind = np.floor(len(train_set_complete) * 0.75)
    ind = 1
    for alf in np.linspace(min_speed, max_speed, num=np.floor((max_speed - min_speed) / step_size)):
        print(alf)
        alfa = alf
        nn.init(alfa_=alfa, seed=123)
        train_set = train_set_complete['Filename'][0:ind]
        lbl_train = (train_set_complete[['Upper_left_corner_X', 'Upper_left_corner_Y',
                                         'Lower_right_corner_X', 'Lower_right_corner_Y']][0:ind])

        print("Learn")
        for i in range(0, ind, batch_size):
            imgs, lbls = prepare_images.prepare(dataset_path + train_set[i].decode('utf8'), lbl_train[i])
            nn.learning(x_in=imgs, lbl_in=lbls)

        print("Predict")
        # predict_res = []
        # for test in test_set:
        #     predict_res.append(nn.predict(test))
        # s = sum(predict_res == lbl_test)
        # proc_n = s / len(lbl_test)
        # print(proc_n)
        # res.append({
        #     'training_size': ind,
        #     'accuracy': proc_n,
        #     'learning_speed': alf,
        #     'hidden_layer_size': hl,
        #     'time': m_time
        # })
    # write_results(res, "training_speed_measurements")
    return res


def test_all():
    print("Test learning speed")
    test_learning_speed(0.5, 3, 0.5, init=True)
