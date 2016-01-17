import csv
import datetime as dt
import os
import timeit

import numpy as np
import numpy.lib.recfunctions

import nn


def test_init(seed=16):
    global train_set_complete

    dataset_path = "h:/_diplomaData/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
    annotation_path = dataset_path + 'allAnnotations.csv'
    negatives_path = dataset_path + "negatives.dat"
    rnd = np.random.RandomState(seed)

    # Read data from files
    image_data = np.genfromtxt(annotation_path, delimiter=';', names=True, dtype=None)
    negatives = np.genfromtxt(negatives_path, dtype=None)
    train_set_complete = numpy.lib.recfunctions.stack_arrays((image_data, negatives), autoconvert=True, usemask=False)

    # Get random
    np.random.shuffle(train_set_complete)


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


# test for train set size
def test_learning_size(min_size=500, max_size=5000, step_size=500, init=False):
    # I don't want to do it multiply times, read large file is long.
    if not init:
        test_init()

    res = []
    for ind in range(min_size, max_size, step_size):
        print(ind)
        hl = 25
        alfa = 1
        nn.init(input_size=28 * 28, hidden_size=hl, output_size=10, alfa_=alfa, seed=123)

        train_set = train_set_complete[0:ind]

        lbl_train = train_set[:, 0]
        train_set = train_set[:, 1:]

        print("Learn")
        m_time = timeit.timeit(lambda: nn.learning(x_in=train_set, lbl_in=lbl_train), number=1)
        # nn.learning(x_in=train_set, lbl_in=lbl_train)
        print("Predict")
        predict_res = []
        for test in test_set:
            predict_res.append(nn.predict(test))
        s = sum(predict_res == lbl_test)
        proc_n = s / len(lbl_test)
        print(proc_n)
        res.append({
            'training_size': ind,
            'accuracy': proc_n,
            'learning_speed': alfa,
            'hidden_layer_size': hl,
            'time': m_time
        })
    write_results(res, "size_measurements")
    return res


# test for train speed
def test_learning_speed(min_speed=0.1, max_speed=2, step_size=0.1, init=False):
    # I don't want to do it multiply times, read large file is long.
    if not init:
        test_init()

    res = []
    ind = np.floor(len(train_set_complete) * 0.75)
    for alf in np.linspace(min_speed, max_speed, num=np.floor((max_speed - min_speed) / step_size)):
        print(alf)
        alfa = alf
        nn.init(alfa_=alfa, seed=123)
        train_set = train_set_complete['Filename'][0:ind]
        lbl_train = (train_set_complete['Upper_left_corner_X'][0:ind],
                     train_set_complete['Lower_right_corner_X'][0:ind],
                     train_set_complete['Upper_left_corner_Y'][0:ind],
                     train_set_complete['Lower_right_corner_Y'][0:ind]
                     )

        print("Learn")
        m_time = timeit.timeit(lambda: nn.learning(x_in=train_set, lbl_in=lbl_train), number=1)
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
    print("Test learning size")
    test_learning_size(3500, 7000, 500)
    print("Test learning speed")
    test_learning_speed(0.5, 3, 0.5, init=True)
