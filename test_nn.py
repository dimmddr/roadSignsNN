import csv
import datetime as dt
import os
import timeit

import numpy as np

import nn


def test_init(seed = 16):
    global train_set_complete
    global test_set
    global lbl_test
    train = np.loadtxt('train.csv', delimiter=',', skiprows=1, dtype=int)
    np.random.seed(seed)
    l = len(train)
    train_ind = np.random.choice(range(l), replace = False, size = np.floor(l * 0.75))
    train_set_complete = train[train_ind, :]
    test_ind = np.delete(np.array(range(l)), train_ind)
    test_set = train[test_ind, :]
    lbl_test = test_set[:, 0]
    test_set = np.delete(test_set, 0, 1)
    lbl_test[lbl_test == 0] = 10


def write_results(result: list, test_name):
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
    os.chdir(test_path)
    with open(name + ".csv", "wb") as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames= result[0].keys())
        writer.writeheader()
        for res in result:
            writer.writerow(res)
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
        nn.init(input_size = 28 * 28, hidden_size = hl, output_size = 10, alfa_ = alfa, seed = 123)
        
        train_set = train_set_complete[0:ind]
        
        lbl_train = train_set[:, 0]
        train_set = train_set[:, 1:]
        lbl_train[lbl_train == 0] = 10
        
        print("Learn")
        m_time = timeit.timeit(nn.learning(x_in=train_set, lbl_in=lbl_train))
        print("Predict")
        predict_res = []
        for test in test_set:
            predict_res.append(nn.predict(test))
        s = sum(predict_res == lbl_test)
        proc_n = s/len(lbl_test)
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


# test for hidden layer size
def test_hidden_layer_size(min_size=15, max_size=50, step_size=5, init=False):
    # I don't want to do it multiply times, read large file is long.
    if not init:
        test_init()

    res = []
    for hl in range(min_size, max_size, step_size):
        print(hl)
        ind = 3500
        alfa = 1
        nn.init(input_size=28 * 28, hidden_size=hl, output_size=10, alfa_=alfa, seed=123)

        train_set = train_set_complete[0:ind]

        lbl_train = train_set[:, 0]
        train_set = train_set[:, 1:]
        lbl_train[lbl_train == 0] = 10

        print("Learn")
        m_time = timeit.timeit(nn.learning(x_in=train_set, lbl_in=lbl_train))
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
    write_results(res, "hidden_layer_size_measurements")
    return res


# test for train speed
def test_learning_speed(min_speed=0.1, max_speed=2, step_size=0.1, init=False):
    # I don't want to do it multiply times, read large file is long.
    if not init:
        test_init()

    res = []
    for alf in range(min_speed, max_speed, step_size):
        print(alf)
        ind = 3500
        alfa = alf
        hl = 25
        nn.init(input_size=28 * 28, hidden_size=hl, output_size=10, alfa_=alfa, seed=123)

        train_set = train_set_complete[0:ind]

        lbl_train = train_set[:, 0]
        train_set = train_set[:, 1:]
        lbl_train[lbl_train == 0] = 10

        print("Learn")
        m_time = timeit.timeit(nn.learning(x_in=train_set, lbl_in=lbl_train))
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
            'learning_speed': alf,
            'hidden_layer_size': hl,
            'time': m_time
        })
    write_results(res, "training_speed_measurements")
    return res


def test_all():
    print("Test learning size")
    test_learning_size(500, 7000, 500)
    print("Test hidden layer size")
    test_hidden_layer_size(15, 50, 5)
    print("Test learning speed")
    test_learning_speed(0.1, 3, 0.1)
